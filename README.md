# Video Game Sales — Hit Prediction

## Question
Can a game's platform, genre, and publisher predict whether it becomes a "hit" (sells over 1 million copies globally)?

## Data
Kaggle "Video Game Sales" dataset — ~16,500 games with platform, genre, publisher, and regional/global sales figures.

## Approach
- Cleaned the dataset by dropping rows with missing values
- Created a binary target: is_hit = 1 if Global_Sales > 1M copies, else 0
- Converted Platform and Genre into numeric features using one-hot encoding (pd.get_dummies)
- Trained and compared three model types — Decision Tree, Random Forest, and Logistic Regression — on an 80/20 train-test split
- Tested whether adding Publisher (grouped into the top 10 publishers + "Other") improved predictions
- Used feature importance analysis to understand which features the best model actually relied on

## Key Findings

**1. Baseline comparison:** All three models landed within 0.5 percentage points of each other (88.4%–88.8% accuracy), only marginally above the dataset's majority-class baseline of 87.5% (the accuracy of simply always guessing "not a hit"). The fact that three structurally different model types converge to the same ceiling suggests the limitation lies in the features available, not the choice of algorithm.

**2. Publisher didn't help, despite appearing important:** Adding Publisher as a feature left accuracy essentially unchanged (±0.1%) across all three models. However, feature importance analysis on the Decision Tree showed `Publisher_Nintendo` as by far the single most relied-upon feature (47% of total importance). The likely explanation: Nintendo games cluster heavily on Nintendo's own platforms (Wii, DS, NES, GB), which the model could already identify through the Platform feature alone — so Publisher mostly duplicates information the model already had, rather than adding new predictive signal.

**3. Conclusion:** Platform, Genre, and Publisher together only weakly predict commercial hit status. Sales success likely depends more on factors not captured here — such as marketing budget, review scores, or franchise/sequel effects.

## What I'd try next
- Test whether Year or franchise/sequel status adds real signal beyond what Platform already captures
- Try feature combinations (e.g. Platform × Genre interactions) rather than treating each independently
- Source review-score data to test whether game quality predicts sales better than publisher or platform alone
