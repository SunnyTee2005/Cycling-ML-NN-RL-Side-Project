from cycling_coach.ingest import load_activity_index

CSV = (
    "Activity ID,Activity Date,Activity Name,Activity Type,Elapsed Time,Distance,"
    "Commute,Activity Gear,Filename,Elapsed Time,Moving Time,Distance,Weighted Average Power\n"
    '2,"Mar 7, 2023, 8:20:14 AM",Ride B,Ride,100,1.00,false,TCR BLACK,activities/2.fit.gz,100,90,1000.0,150\n'
    '1,"Mar 6, 2023, 8:20:14 AM",Ride A,Ride,100,2.00,true,Reacto 400,activities/1.gpx,100,90,2000.0,\n'
    '3,"Mar 8, 2023, 8:20:14 AM",Run,Run,100,3.00,false,,activities/3.fit.gz,100,90,3000.0,\n'
    '4,"Mar 9, 2023, 8:20:14 AM",Manual,Ride,100,4.00,false,,,100,90,4000.0,\n'
)


def test_load_activity_index(tmp_path):
    path = tmp_path / "activities.csv"
    path.write_text(CSV)
    df = load_activity_index(path)

    assert df["activity_id"].tolist() == [1, 2]  # rides with files only, sorted by time
    assert df["distance_m"].tolist() == [2000.0, 1000.0]  # metres column, not km
    assert df["has_power_meter"].tolist() == [False, True]
    assert str(df["start_time"].dt.tz) == "UTC"


def test_crop_to_activity_keeps_only_window():
    import pandas as pd

    from cycling_coach.ingest import crop_to_activity

    t0 = pd.Timestamp("2024-02-03 01:00:00", tz="UTC")
    df = pd.DataFrame({"timestamp": pd.date_range(t0, periods=100, freq="s")})
    out = crop_to_activity(df, t0 + pd.Timedelta(seconds=10), elapsed_s=20)
    assert len(out) == 21
    assert out["timestamp"].iloc[0] == t0 + pd.Timedelta(seconds=10)
