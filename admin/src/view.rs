//! Display formatting shared by all pages. Templates get pre-formatted strings.

const MINUS: char = '−';

fn group_thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn abs_money(v: f64) -> String {
    let a = v.abs();
    if a < 10.0 && a.fract().abs() > f64::EPSILON {
        format!("€{a:.2}")
    } else {
        format!("€{}", group_thousands(a.round() as u64))
    }
}

pub fn money(v: f64) -> String {
    if v < 0.0 {
        format!("{MINUS}{}", abs_money(v))
    } else {
        abs_money(v)
    }
}

pub fn signed_money(v: f64) -> String {
    if v < 0.0 {
        format!("{MINUS}{}", abs_money(v))
    } else {
        format!("+{}", abs_money(v))
    }
}

pub fn pct(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.0}%")).unwrap_or_else(|| "—".into())
}

pub fn signed_pct(v: Option<f64>) -> String {
    match v {
        Some(x) if x < 0.0 => format!("{MINUS}{:.1}%", x.abs()),
        Some(x) => format!("+{x:.1}%"),
        None => "—".into(),
    }
}

pub fn pnl_class(v: f64) -> &'static str {
    if v > 0.0 {
        "pos"
    } else if v < 0.0 {
        "neg"
    } else {
        ""
    }
}

pub fn age(secs: f64) -> String {
    let s = secs.max(0.0) as u64;
    match s {
        0..=59 => format!("{s}s"),
        60..=3599 => format!("{}m", s / 60),
        3600..=172_799 => format!("{}h", s / 3600),
        _ => format!("{}d", s / 86_400),
    }
}

pub fn ago(secs: f64) -> String {
    format!("{} ago", age(secs))
}

pub fn short_wallet(w: &str) -> String {
    let c: Vec<char> = w.chars().collect();
    if c.len() <= 8 {
        return w.to_string();
    }
    format!(
        "{}…{}",
        c[..4].iter().collect::<String>(),
        c[c.len() - 2..].iter().collect::<String>()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn money_formats_thousands() {
        assert_eq!(money(7412.4), "€7,412");
        assert_eq!(money(-5472.0), "−€5,472");
        assert_eq!(money(0.0), "€0");
    }

    #[test]
    fn signed_money_always_has_sign() {
        assert_eq!(signed_money(42.0), "+€42");
        assert_eq!(signed_money(-188.4), "−€188");
        assert_eq!(signed_money(3.5), "+€3.50");
    }

    #[test]
    fn pct_and_signed_pct() {
        assert_eq!(pct(Some(47.04)), "47%");
        assert_eq!(pct(None), "—");
        assert_eq!(signed_pct(Some(-10.43)), "−10.4%");
        assert_eq!(signed_pct(Some(2.0)), "+2.0%");
    }

    #[test]
    fn pnl_class_by_sign() {
        assert_eq!(pnl_class(1.0), "pos");
        assert_eq!(pnl_class(-0.01), "neg");
        assert_eq!(pnl_class(0.0), "");
    }

    #[test]
    fn ago_and_age_units() {
        assert_eq!(ago(4.0), "4s ago");
        assert_eq!(ago(180.0), "3m ago");
        assert_eq!(ago(7200.0), "2h ago");
        assert_eq!(ago(200_000.0), "2d ago");
        assert_eq!(age(10_800.0), "3h");
        assert_eq!(age(90.0), "1m");
    }

    #[test]
    fn short_wallet_truncates() {
        assert_eq!(short_wallet("0x9f12345678a1"), "0x9f…a1");
        assert_eq!(short_wallet("abc"), "abc");
        // multi-byte input must not panic on a char boundary
        assert_eq!(short_wallet("добрыйдень-трейдер"), "добр…ер");
    }
}
