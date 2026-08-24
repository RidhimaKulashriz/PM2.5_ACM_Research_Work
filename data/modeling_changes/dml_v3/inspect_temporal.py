from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[3]
train = pd.read_csv(root / 'data/modeling_changes/splits/train.csv')
test = pd.read_csv(root / 'data/modeling_changes/splits/test.csv')
for name, df in [('train', train), ('test', test)]:
    print(f'[{name}] rows={len(df)} stations={df.station.nunique()}')
    print('columns=', [c for c in ['station','year','month','date','pm25','sentinel2_ndvi_mean_1000m','sentinel2_ndvi_mean_500m','modis_ndvi_mean_1000m'] if c in df.columns])
    print('years=', sorted(df.year.dropna().unique().tolist()) if 'year' in df else 'missing')
    print('months=', sorted(df.month.dropna().unique().tolist()) if 'month' in df else 'missing')
    if 'date' in df.columns:
        print('date_minmax=', df['date'].min(), df['date'].max())
    else:
        period = pd.to_datetime(dict(year=df['year'].astype(int), month=df['month'].astype(int), day=1))
        print('period_minmax=', period.min(), period.max())
    print('duplicate_station_month=', int(df.duplicated(['station','year','month']).sum()))
    print('rows_per_station_quantiles=', df.groupby('station').size().quantile([0,.25,.5,.75,1]).to_dict())

all_df = pd.concat([train.assign(split='train'), test.assign(split='test')], ignore_index=True)
all_df['period'] = pd.to_datetime(dict(year=all_df.year.astype(int), month=all_df.month.astype(int), day=1))
all_df = all_df.sort_values(['station','period'])
for treatment in ['sentinel2_ndvi_mean_1000m','sentinel2_ndvi_mean_500m','modis_ndvi_mean_1000m']:
    all_df[f'lag_{treatment}'] = all_df.groupby('station')[treatment].shift(1)
    print(f'lag_{treatment}_nonmissing={int(all_df[f"lag_{treatment}"].notna().sum())}')
    print(f'lag_{treatment}_same_station_order_check={bool((all_df.groupby("station")["period"].diff().dropna() > pd.Timedelta(0)).all())}')
print('cross_split_temporal_overlap=', int(all_df.groupby(['station','period']).split.nunique().gt(1).sum()))
print('station_count=', all_df.station.nunique())
