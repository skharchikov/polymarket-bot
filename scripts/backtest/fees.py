"""Fee-accurate realized PnL for one resolved bet.

Matches live accounting in common/src/storage/postgres.rs:1013-1018:
- 2% entry fee on the stake,
- 2% exit fee on the gross payout,
- a win pays shares (= stake / bet_price) at $1 each.
"""


def net_pnl(stake, bet_price, won, entry_fee_rate=0.02, exit_fee_rate=0.02):
    """Realized PnL for one resolved bet.

    `bet_price` is the price paid on the bet's own side (for a NO bet, pass the
    NO price, i.e. 1 - yes_price). On a loss the whole stake plus entry fee is
    lost (a zero payout incurs no exit fee).
    """
    if bet_price <= 0.0 or bet_price >= 1.0 or stake <= 0.0:
        return 0.0
    entry_fee = stake * entry_fee_rate
    if won:
        payout = stake / bet_price          # shares * $1
        exit_fee = payout * exit_fee_rate
        return payout - exit_fee - stake - entry_fee
    return -(stake + entry_fee)
