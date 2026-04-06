import duckdb, glob

d = "/mnt/HC_Volume_105339755/data/polymarket"

for label, subdir in [("Markets", "markets"), ("Trades", "trades")]:
    files = sorted(glob.glob(f"{d}/{subdir}/*.parquet"))
    total = 0
    good = 0
    closed = 0
    for f in files:
        try:
            con = duckdb.connect()
            r = con.execute(f"SELECT COUNT(*) FROM '{f}'").fetchone()
            total += r[0]
            good += 1
            if label == "Markets":
                c = con.execute(f"SELECT COUNT(*) FILTER (WHERE closed) FROM '{f}'").fetchone()
                closed += c[0]
            con.close()
        except Exception as e:
            print(f"  skip: {f.split('/')[-1]} ({e})")
    print(f"{label}: {good}/{len(files)} files ok, {total:,} rows" + (f", {closed:,} closed" if label == "Markets" else ""))
