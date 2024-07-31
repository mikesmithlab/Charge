import pandas as pd


pd.read_csv()


df = pd.DataFrame({'time':[1,2,3], 'channelA':[1,2,3]})
df['charge'] = df['channelA']*2
df.to_csv('test.csv', index=False)
