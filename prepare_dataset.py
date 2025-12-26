import pandas as pd

# 1. Read your current CSV (3 columns, no header)
df = pd.read_csv("data/product_reviews.csv",
                 header=None,
                 names=["rating", "title", "review"])

# 2. Create sentiment from rating
#    4 or 5 -> Positive, 1 or 2 -> Negative, 3 -> Neutral
def rating_to_sentiment(r):
    try:
        r = float(r)
    except:
        return "Neutral"
    if r >= 4:
        return "Positive"
    elif r <= 2:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["rating"].apply(rating_to_sentiment)

# 3. Keep only the columns the project needs, with correct names
df_out = df[["review", "sentiment"]]

# 4. Overwrite the file in the exact format the project expects
df_out.to_csv("data/product_reviews.csv", index=False)
print("Prepared dataset saved to data/product_reviews.csv")