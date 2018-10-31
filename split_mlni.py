import pandas as pd

df = pd.read_csv("mnli_val.tsv", sep="\t")
print(len(df))  # 5000

print(df.genre.unique())
"""
array(['fiction', 'telephone', 'slate', 'government', 'travel'],
      dtype=object)
"""

df_fiction = df.loc[(df.genre == "fiction")]
print(len(df_fiction))  # 995

df_telephone = df.loc[(df.genre == "telephone")]
print(len(df_telephone))  # 1005

df_slate = df.loc[(df.genre == "slate")]
print(len(df_slate))  # 1002

df_government = df.loc[(df.genre == "government")]
print(len(df_government))  # 1016

df_travel = df.loc[(df.genre == "travel")]
print(len(df_travel))  # 982

df_fiction.drop(columns=["genre"], inplace=True)
df_telephone.drop(columns=["genre"], inplace=True)
df_slate.drop(columns=["genre"], inplace=True)
df_government.drop(columns=["genre"], inplace=True)
df_travel.drop(columns=["genre"], inplace=True)

df_fiction.to_csv("mnli_val.fiction.tsv", sep="\t", encoding="utf-8", index=False)
df_telephone.to_csv("mnli_val.telephone.tsv", sep="\t", encoding="utf-8", index=False)
df_slate.to_csv("mnli_val.slate.tsv", sep="\t", encoding="utf-8", index=False)
df_government.to_csv("mnli_val.government.tsv", sep="\t", encoding="utf-8", index=False)
df_travel.to_csv("mnli_val.travel.tsv", sep="\t", encoding="utf-8", index=False)
