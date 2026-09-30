//! Minimal line-based editing of block-style YAML mappings.
//!
//! Open-AgentRL's configs carry their setup instructions as inline comments,
//! so values are replaced in place rather than round-tripping through a YAML
//! library, which would drop every comment.

/// A parsed `key: value  # comment` line.
struct Line<'a> {
    indent: usize,
    key: &'a str,
    value: &'a str,
    comment: &'a str,
}

fn parse_line(line: &str) -> Option<Line<'_>> {
    let trimmed = line.trim_start();
    if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('-') {
        return None;
    }
    let indent = line.len() - trimmed.len();
    let colon = trimmed.find(':')?;
    let key = &trimmed[..colon];
    if key.is_empty()
        || !key
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return None;
    }
    let rest = &trimmed[colon + 1..];
    let (value, comment) = split_comment(rest);
    Some(Line {
        indent,
        key,
        value: value.trim(),
        comment,
    })
}

/// Splits off a trailing `# comment`, ignoring `#` inside quotes.
fn split_comment(s: &str) -> (&str, &str) {
    let mut quote: Option<char> = None;
    let mut prev_space = true;
    for (i, c) in s.char_indices() {
        match (quote, c) {
            (Some(q), c) if c == q => quote = None,
            (None, '"' | '\'') => quote = Some(c),
            (None, '#') if prev_space => return (&s[..i], s[i..].trim_end()),
            _ => {}
        }
        prev_space = c.is_whitespace();
    }
    (s, "")
}

fn find(lines: &[&str], path: &[&str]) -> Option<usize> {
    let mut stack: Vec<(usize, &str)> = Vec::new();
    for (i, raw) in lines.iter().enumerate() {
        let Some(line) = parse_line(raw) else {
            continue;
        };
        while stack.last().is_some_and(|&(ind, _)| ind >= line.indent) {
            stack.pop();
        }
        stack.push((line.indent, line.key));
        if stack.len() == path.len() && stack.iter().zip(path).all(|((_, k), p)| k == p) {
            return Some(i);
        }
    }
    None
}

/// Returns the scalar at a dotted path such as `model.policy_model`, with
/// quotes removed. Empty values, `null` and `~` count as unset.
pub fn get(text: &str, path: &str) -> Option<String> {
    let path: Vec<&str> = path.split('.').collect();
    let lines: Vec<&str> = text.lines().collect();
    let line = parse_line(lines[find(&lines, &path)?])?;
    let v = unquote(line.value);
    if v.is_empty() || v == "null" || v == "~" {
        None
    } else {
        Some(v.to_string())
    }
}

/// Replaces the value at a dotted path, keeping indentation and any comment.
/// `raw` is written verbatim; use [`quote`] for strings.
pub fn set(text: &str, path: &str, raw: &str) -> Result<String, String> {
    let parts: Vec<&str> = path.split('.').collect();
    let lines: Vec<&str> = text.lines().collect();
    let idx = find(&lines, &parts).ok_or_else(|| format!("key `{path}` not found"))?;
    let line = parse_line(lines[idx]).expect("found line parses");
    let mut new = format!("{}{}: {}", " ".repeat(line.indent), line.key, raw);
    if !line.comment.is_empty() {
        new.push(' ');
        new.push_str(line.comment);
    }
    let mut out: Vec<String> = lines.iter().map(|l| l.to_string()).collect();
    out[idx] = new;
    let mut joined = out.join("\n");
    if text.ends_with('\n') {
        joined.push('\n');
    }
    Ok(joined)
}

pub fn quote(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

fn unquote(s: &str) -> &str {
    for q in ['"', '\''] {
        if s.len() >= 2 && s.starts_with(q) && s.ends_with(q) {
            return &s[1..s.len() - 1];
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
wandb:
  entity: null

experiment:
    project: \"coding_rl\" # need to be the same as the file name
    num_node: 4 # the number of machines you have

system:
    HF_HOME: # absolute path of your HF_HOME
    env_name: \"rlanything\" # env name
    TYPES: # list
        - a.b
        - c.d
    rl_base_dir: # /absolute/path/to/Open-AgentRL

model:
    policy_model: # absolute path of your policy model
";

    #[test]
    fn reads_values() {
        assert_eq!(
            get(SAMPLE, "experiment.project").as_deref(),
            Some("coding_rl")
        );
        assert_eq!(get(SAMPLE, "experiment.num_node").as_deref(), Some("4"));
        assert_eq!(get(SAMPLE, "wandb.entity"), None);
        assert_eq!(get(SAMPLE, "system.HF_HOME"), None);
        assert_eq!(get(SAMPLE, "model.missing"), None);
        // Same key name under a different parent is not confused.
        assert_eq!(get(SAMPLE, "model.env_name"), None);
    }

    #[test]
    fn finds_keys_after_list_items() {
        let out = set(SAMPLE, "system.rl_base_dir", &quote("/x")).unwrap();
        assert_eq!(get(&out, "system.rl_base_dir").as_deref(), Some("/x"));
    }

    #[test]
    fn set_keeps_comments_and_other_lines() {
        let out = set(SAMPLE, "model.policy_model", &quote("/models/p")).unwrap();
        assert!(
            out.contains("    policy_model: \"/models/p\" # absolute path of your policy model\n")
        );
        assert_eq!(out.lines().count(), SAMPLE.lines().count());
        assert!(out.ends_with('\n'));
        let out = set(&out, "experiment.num_node", "1").unwrap();
        assert!(out.contains("    num_node: 1 # the number of machines you have"));
    }

    #[test]
    fn set_unknown_key_errors() {
        assert!(set(SAMPLE, "model.nope", "1").is_err());
    }

    #[test]
    fn hash_inside_quotes_is_not_a_comment() {
        let text = "a:\n  b: \"x # y\" # real\n";
        assert_eq!(get(text, "a.b").as_deref(), Some("x # y"));
    }

    #[test]
    fn quote_escapes() {
        assert_eq!(quote(r#"a"b\c"#), r#""a\"b\\c""#);
    }
}
