pub mod overview;
pub mod status;

/// HTML selects submit "" for "any"; treat that as no filter.
pub fn nonempty(v: Option<String>) -> Option<String> {
    v.filter(|s| !s.trim().is_empty())
}
pub mod history;
pub mod positions;
pub mod signals;
pub mod traders;
