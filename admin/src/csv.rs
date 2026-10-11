use crate::db::history::ResolvedRow;

/// RFC 4180 field; prefixes spreadsheet formula triggers with `'`.
pub fn field(s: &str) -> String {
    let s = if s.starts_with(['=', '+', '-', '@', '\t', '\r']) {
        format!("'{s}")
    } else {
        s.to_string()
    };
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s
    }
}

pub fn write_rows(rows: &[ResolvedRow]) -> String {
    let mut out = String::from(
        "id,placed_at,resolved_at,bot,source,strategy,category,question,side,entry_price,cost,fee_paid,won,pnl\n",
    );
    for r in rows {
        let cols = [
            r.id.to_string(),
            r.placed_at.to_rfc3339(),
            r.resolved_at.map(|t| t.to_rfc3339()).unwrap_or_default(),
            r.bot.clone(),
            r.source.clone(),
            r.strategy.clone(),
            r.category.clone().unwrap_or_default(),
            r.question.clone(),
            r.side.clone(),
            format!("{:.4}", r.entry_price),
            format!("{:.2}", r.cost),
            format!("{:.2}", r.fee_paid),
            r.won.map(|w| w.to_string()).unwrap_or_default(),
            r.pnl.map(|p| format!("{p:.2}")).unwrap_or_default(),
        ];
        // Numeric columns bypass `field` so negative values aren't `'`-prefixed.
        let numeric = [0usize, 9, 10, 11, 13];
        let line: Vec<String> = cols
            .iter()
            .enumerate()
            .map(|(i, c)| {
                if numeric.contains(&i) {
                    c.clone()
                } else {
                    field(c)
                }
            })
            .collect();
        out.push_str(&line.join(","));
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::field;

    #[test]
    fn quotes_only_when_needed() {
        assert_eq!(field("plain"), "plain");
        assert_eq!(field("a,b"), "\"a,b\"");
        assert_eq!(field("say \"hi\""), "\"say \"\"hi\"\"\"");
        assert_eq!(field("line\nbreak"), "\"line\nbreak\"");
    }

    #[test]
    fn neutralises_formula_injection() {
        assert_eq!(field("=HYPERLINK(\"x\")"), "\"'=HYPERLINK(\"\"x\"\")\"");
        assert_eq!(field("+1"), "'+1");
        assert_eq!(field("\tx"), "'\tx");
        assert_eq!(field("\rx"), "\"'\rx\"");
    }
}
