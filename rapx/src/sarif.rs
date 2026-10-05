//! SARIF 2.1.0 export for `rapx check` findings (`--format sarif`).
//!
//! Findings are recorded at the two places the check detectors report —
//! SafeDrop's bug records and RCanary's memory-leak verdict — into a
//! thread-local store, and drained once per analyzed crate by
//! `start_analyzer`, so one complete SARIF log is emitted instead of the
//! human-rendered records. The mapping follows the external dataset
//! generator in the lockbud-stable workspace (`detector-toys/`): one rule
//! per bug kind, lock-type-free properties, and the same findings the text
//! output reports (recording happens on exactly the paths that also print).

use std::cell::RefCell;

use rustc_middle::ty::TyCtxt;
use rustc_span::Span;
use serde_json::{json, Value};

use crate::cli::ReportFormat;

const SARIF_SCHEMA: &str = "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json";

thread_local! {
    /// Findings recorded while the current crate is analyzed. `Span` is
    /// `Copy` and lives as long as the compilation, so raw spans can be
    /// recorded at the detection site and resolved at emission time.
    static FINDINGS: RefCell<Vec<RawFinding>> = const { RefCell::new(Vec::new()) };
}

/// Whether the run emits SARIF instead of the human-rendered records.
/// The detectors are deep in the analysis and print where they detect, so
/// the mode is a flag the print sites consult rather than a parameter
/// threaded through every visitor: in SARIF mode findings are still
/// recorded, but the human records are suppressed, matching the other
/// adapters' "SARIF instead of the raw output" contract.
static SARIF_MODE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_sarif_mode(on: bool) {
    SARIF_MODE.store(on, std::sync::atomic::Ordering::SeqCst);
}

pub fn sarif_mode() -> bool {
    SARIF_MODE.load(std::sync::atomic::Ordering::SeqCst)
}

/// A finding exactly as the detectors record it.
#[derive(Clone, Debug)]
pub struct RawFinding {
    /// Bug kind: `DanglingPointer`, `UseAfterFree`, `DoubleFree`,
    /// `UseAfterFreeAndDoubleFree`, or `MemoryLeak`.
    pub kind: &'static str,
    pub containing_function: String,
    pub span: Span,
    /// Related spans (e.g. the leak-candidate annotations), resolved into
    /// `relatedLocations`.
    pub related_spans: Vec<Span>,
    pub confidence: Option<usize>,
    /// Whether the finding was recorded on a path that can also run while
    /// unwinding (SafeDrop's `_unwind` bug sets).
    pub during_unwind: bool,
}

pub fn record_finding(
    kind: &'static str,
    containing_function: String,
    span: Span,
    related_spans: Vec<Span>,
    confidence: Option<usize>,
    during_unwind: bool,
) {
    FINDINGS.with(|f| {
        f.borrow_mut().push(RawFinding {
            kind,
            containing_function,
            span,
            related_spans,
            confidence,
            during_unwind,
        })
    });
}

pub fn clear_findings() {
    FINDINGS.with(|f| f.borrow_mut().clear());
}

/// Take the recorded findings, leaving the store empty.
pub fn take_findings() -> Vec<RawFinding> {
    FINDINGS.with(|f| std::mem::take(&mut *f.borrow_mut()))
}

/// A finding with its spans resolved into SARIF regions. Region values are
/// 1-based and INCLUSIVE (SARIF convention); `end_col` is omitted when the
/// span ends exactly at a line boundary (the region then extends to the
/// end of the line).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedFinding {
    pub kind: String,
    pub containing_function: String,
    pub uri: String,
    pub start_line: u32,
    pub start_col: u32,
    pub end_line: u32,
    pub end_col: Option<u32>,
    pub related: Vec<(String, u32, u32, u32, Option<u32>)>,
    pub confidence: Option<usize>,
    pub during_unwind: bool,
}

fn resolve_span(source_map: &rustc_span::source_map::SourceMap, span: Span) -> (String, u32, u32, u32, Option<u32>) {
    let start = source_map.lookup_char_pos(span.lo());
    let end = source_map.lookup_char_pos(span.hi());
    // span.hi() points one past the last character, so its 0-based column
    // equals the 1-based inclusive column of the last character; a column
    // of 0 means the span ends at a line boundary — omit endColumn.
    let (end_line, end_col) = if end.col.0 == 0 && end.line > start.line {
        (end.line - 1, None)
    } else {
        (end.line, Some(end.col.0 as u32))
    };
    let start_col = start.col.0 as u32 + 1;
    // Zero-width spans can make the inclusive end column precede the start
    // column; clamp so the region stays well-formed.
    let end_col = end_col.map(|c| c.max(start_col));
    let uri = match &start.file.name {
        rustc_span::FileName::Real(real) => real
            .local_path()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| format!("{:?}", real)),
        other => format!("{:?}", other),
    };
    (uri, start.line as u32, start_col, end_line as u32, end_col)
}

/// Resolve recorded findings against the compilation's source map.
pub fn resolve_findings<'tcx>(tcx: TyCtxt<'tcx>, findings: &[RawFinding]) -> Vec<ResolvedFinding> {
    let source_map = tcx.sess.source_map();
    findings
        .iter()
        .map(|f| {
            let (uri, start_line, start_col, end_line, end_col) =
                resolve_span(&source_map, f.span);
            let related = f
                .related_spans
                .iter()
                .map(|s| resolve_span(&source_map, *s))
                .collect();
            ResolvedFinding {
                kind: f.kind.to_owned(),
                containing_function: f.containing_function.clone(),
                uri,
                start_line,
                start_col,
                end_line,
                end_col,
                related,
                confidence: f.confidence,
                during_unwind: f.during_unwind,
            }
        })
        .collect()
}

fn region(f: &ResolvedFinding) -> Value {
    let mut region = json!({ "startLine": f.start_line, "startColumn": f.start_col });
    if f.end_line != f.start_line {
        region["endLine"] = json!(f.end_line);
    }
    if let Some(end_col) = f.end_col {
        region["endColumn"] = json!(end_col);
    }
    region
}

fn location(uri: &str, region: &Value) -> Value {
    json!({ "physicalLocation": {
        "artifactLocation": { "uri": uri },
        "region": region,
    }})
}

/// The rule identity and description for a bug kind.
fn rule_for(kind: &str) -> (String, String, Vec<&'static str>) {
    let (id, desc, cwe): (&str, &str, Vec<&'static str>) = match kind {
        "DanglingPointer" => (
            "dangling-pointer",
            "Dangling pointer created or returned",
            vec!["CWE-416"],
        ),
        "UseAfterFree" => (
            "use-after-free",
            "Pointer used after the pointee was freed",
            vec!["CWE-416"],
        ),
        "DoubleFree" => (
            "double-free",
            "Allocation freed twice",
            vec!["CWE-415"],
        ),
        "UseAfterFreeAndDoubleFree" => (
            "use-after-free-and-double-free",
            "Pointer use after free combined with a double free",
            vec!["CWE-416", "CWE-415"],
        ),
        "MemoryLeak" => (
            "memory-leak",
            "Heap allocation that is never freed on any path",
            vec!["CWE-401"],
        ),
        _ => ("other", "Other finding", vec![]),
    };
    (format!("rapx/{}", id), desc.to_owned(), cwe)
}

/// Build one complete SARIF 2.1.0 log for one analyzed crate. Pure over
/// resolved findings so it is unit-testable without a compiler.
pub fn build_sarif_log(crate_name: &str, findings: &[ResolvedFinding]) -> Value {
    let mut results: Vec<Value> = Vec::new();
    let mut rules: Vec<Value> = Vec::new();
    let mut seen_kinds: Vec<String> = Vec::new();
    let mut artifacts: Vec<Value> = Vec::new();
    let mut seen_uris: Vec<String> = Vec::new();

    for f in findings {
        let (rule_id, desc, cwe) = rule_for(&f.kind);
        if !seen_kinds.iter().any(|k| k == &f.kind) {
            seen_kinds.push(f.kind.clone());
            let mut rule = json!({
                "id": rule_id,
                "name": f.kind,
                "shortDescription": { "text": desc },
                "fullDescription": { "text": format!(
                    "rapx check finding ({})", f.kind
                )},
            });
            if !cwe.is_empty() {
                rule["properties"] = json!({ "cwe": cwe });
            }
            rules.push(rule);
        }
        for uri in std::iter::once(&f.uri).chain(f.related.iter().map(|r| &r.0)) {
            if !seen_uris.iter().any(|u| u == uri) {
                seen_uris.push(uri.clone());
                artifacts.push(json!({ "location": { "uri": uri } }));
            }
        }
        let related: Vec<Value> = f
            .related
            .iter()
            .map(|(uri, start_line, start_col, end_line, end_col)| {
                let mut region = json!({ "startLine": start_line, "startColumn": start_col });
                if end_line != start_line {
                    region["endLine"] = json!(end_line);
                }
                if let Some(end_col) = end_col {
                    region["endColumn"] = json!(end_col);
                }
                location(uri, &region)
            })
            .collect();
        let mut properties = json!({
            "detector": "rapx",
            "containingFunction": f.containing_function,
        });
        if let Some(confidence) = f.confidence {
            properties["confidence"] = json!(confidence);
        }
        if f.during_unwind {
            properties["duringUnwinding"] = json!(true);
        }
        let mut result = json!({
            "ruleId": rule_id,
            "level": "warning",
            "message": { "text": format!(
                "{} in function {}", f.kind, f.containing_function
            )},
            "locations": [location(&f.uri, &region(f))],
            "properties": properties,
        });
        if !related.is_empty() {
            result["relatedLocations"] = Value::Array(related);
        }
        results.push(result);
    }

    json!({
        "$schema": SARIF_SCHEMA,
        "version": "2.1.0",
        "runs": [{
            "tool": { "driver": {
                "name": "rapx",
                "version": env!("CARGO_PKG_VERSION"),
                "informationUri": "https://github.com/safer-rust/RAPx",
                "rules": rules,
            }},
            "results": results,
            "artifacts": artifacts,
            "automationDetails": { "id": format!("rapx-scan/{}", crate_name) },
            "invocations": [{ "executionSuccessful": true }],
            "properties": {
                "columnConvention": "1-based lines/columns; region endColumn is inclusive; a missing endColumn extends to the end of the line",
            },
        }],
    })
}

/// Drain the store and emit one SARIF log for the analyzed crate. The
/// human-rendered records must not be printed when `format` is Sarif, so
/// this clears the mode flag after emitting.
pub fn emit_if_sarif(tcx: TyCtxt<'_>, crate_name: &str, format: ReportFormat) {
    if format != ReportFormat::Sarif {
        return;
    }
    let findings = resolve_findings(tcx, &take_findings());
    let log = build_sarif_log(crate_name, &findings);
    rap_warn!(
        "{}",
        serde_json::to_string(&log).expect("SARIF log serializes")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finding(kind: &str, fn_name: &str) -> ResolvedFinding {
        ResolvedFinding {
            kind: kind.to_owned(),
            containing_function: fn_name.to_owned(),
            uri: "src/main.rs".to_owned(),
            start_line: 9,
            start_col: 1,
            end_line: 15,
            end_col: Some(2),
            related: vec![("src/main.rs".to_owned(), 13, 5, 13, Some(27))],
            confidence: Some(99),
            during_unwind: false,
        }
    }

    #[test]
    fn test_build_sarif_log_shape() {
        let log = build_sarif_log(
            "uaf_1",
            &[finding("DanglingPointer", "create_vec"), finding("MemoryLeak", "main")],
        );
        assert_eq!(log["version"], "2.1.0");
        let run = &log["runs"][0];
        assert_eq!(run["tool"]["driver"]["name"], "rapx");
        assert_eq!(run["automationDetails"]["id"], "rapx-scan/uaf_1");
        let results = run["results"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["ruleId"], "rapx/dangling-pointer");
        assert_eq!(results[0]["level"], "warning");
        assert_eq!(
            results[0]["message"]["text"],
            "DanglingPointer in function create_vec"
        );
        let region = &results[0]["locations"][0]["physicalLocation"]["region"];
        assert_eq!(region["startLine"], 9);
        assert_eq!(region["endLine"], 15);
        assert_eq!(region["endColumn"], 2);
        // related location (creation site) exported
        let related = results[0]["relatedLocations"].as_array().unwrap();
        assert_eq!(related.len(), 1);
        assert_eq!(related[0]["physicalLocation"]["region"]["startColumn"], 5);
        assert_eq!(results[0]["properties"]["confidence"], 99);
        assert!(results[0]["properties"].get("duringUnwinding").is_none());
        // rules per kind with CWE tags
        let rules = run["tool"]["driver"]["rules"].as_array().unwrap();
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0]["properties"]["cwe"][0], "CWE-416");
        assert_eq!(rules[1]["properties"]["cwe"][0], "CWE-401");
    }

    #[test]
    fn test_merged_kind_has_two_cwes() {
        let log = build_sarif_log(
            "c",
            &[finding("UseAfterFreeAndDoubleFree", "f")],
        );
        let cwe = log["runs"][0]["tool"]["driver"]["rules"][0]["properties"]["cwe"]
            .as_array()
            .unwrap();
        assert_eq!(cwe.len(), 2);
    }

    #[test]
    fn test_empty_findings_produce_empty_results() {
        let log = build_sarif_log("c", &[]);
        assert!(log["runs"][0]["results"].as_array().unwrap().is_empty());
    }
}
