import pandas as pd
from util import no_zeros_formatter

df=pd.read_csv('results.tsv', sep='\t', index_col=0)
df=df.drop(['empathy_bin', 'distress_bin'], 1)
df=df.drop(['maxent'],0)
df['Mean']=df.mean(axis=1)
df=df.rename(columns={'empathy':'Empathy', 'distress':'Distress'})
df=df.rename(index={'cnn':'CNN', 'ffn':'FFN','ridge':'Ridge'})
print(df)
print()



### latex output
print(r'\begin{tabular}{'+'l'+df.shape[0]*'r'+'}')
print(r'\toprule')
header= r' & '+ ' & '.join([r'{\bf '+x+r'}' for x in list(df)]) + r' \\'
print(header)
print(r'\midrule')
for row in df.index:
	numbers=[no_zeros_formatter(x,3) for x in df.loc[row]]

	print(' & '.join([row]+numbers)+r'\\')
print(r'\bottomrule')
print(r'\end{tabular}')

print()

