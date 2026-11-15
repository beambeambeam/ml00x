# Machine Learning Project Instruction

Teams of five will build a Player Intelligence System for Karena Thailand using five machine-learning tasks: binary classification, multi-class classification, regression, image classification, and unsupervised anomaly detection. The tasks address game integrity, player segments, spending, game identification, and account security.

Use the provided Kaggle datasets and follow the data-use and collaboration rules in this guide. Submit predictions for all five tasks, a technical report, and organized source code. The Kaggle leaderboard counts for 70% of the grade, the report for 20%, and source code for 10%.

## Introduction

This project challenges teams of five to build a Player Intelligence System for Karena Thailand, a gaming platform company. Teams will develop five connected machine-learning models: binary classification, multi-class classification, regression, image classification, and unsupervised anomaly detection. The project applies machine learning to business problems in game integrity, player behavior, monetization, and retention.

Teams will use the provided Kaggle datasets and submit predictions for all five tasks. They will also submit a technical report and organized source code.

## Project Setting: Situation

Your team consists of Machine Learning Engineers at Karena Thailand, a gaming platform company similar to Garena. The company develops and publishes multiple online games, for example PUBE Mobile, Free Fried, FiveN, Roblock, and 7 Knights.

## Project Setting: The Challenge

Karena needs an AI-powered Player Intelligence System to:

- Maintain game integrity across all platforms.
- Understand player behavior and preferences.
- Optimize monetization strategies.
- Improve player retention and lifetime value.

## Your Team’s Mission

Build a comprehensive ML pipeline with five tasks. The tasks feed into a final Player Lifetime Value (LTV) Prediction System to help the business make data-driven decisions.

### What You’ll Build – Five Interconnected Models

1. Binary Classification
2. Multi-Class Classification
3. Regression
4. Image Classification
5. Anomaly Detection (Unsupervised)

## Instruction Structure & Competition Rules

### Each Task Instruction

- Situation
- Mission
- Task Category
- Evaluation Metric
- Example Features
- Hints & Best Practices
- Dataset structure for the task, available from the Kaggle download

### Competition Rules

**Data**

- Use provided datasets only. Combine Train+Val for training.
- Do not use the Test set for training. Do not use pseudo-labeling or fine-tuning on the Test set.

**Model Development**

- Task 4 allows pre-trained models. Tasks 1, 2, 3, and 5 train from scratch.
- Feature engineering and ensembling are encouraged.
- AutoML frameworks are prohibited. Hyperparameter tuning tools are allowed.

**Collaboration**

- Work only with your team.
- Concept discussion is OK, but do not share code or predictions.
- All members must contribute meaningfully.

## Task 1: Anti-Cheat Pre-Filter

**Task category:** Binary Classification

### Situation

Karena Thailand receives over 10,000 cheat reports daily across all games, but the manual review team can only handle 500 cases per day. This creates a backlog of 45,000+ pending reports. Many reports are false positives from frustrated players who lost matches fairly, while real cheaters continue to damage the game experience and drive legitimate players away. The current automated system has a 70% false positive rate. Innocent players get flagged and sometimes banned, which leads to customer service problems and permanent player loss. Sophisticated cheaters who use subtle aimbots [1] or wallhacks [2] slip through undetected. Player retention has dropped 15% in competitive game modes due to cheating complaints. This directly impacts revenue by approximately ฿15 million daily. The company needs an intelligent pre-filtering system that can accurately identify high-probability cheaters for priority manual review, reduce the workload, and catch real threats.

[1] **Aimbot:** Cheating software that automatically aims at enemies.  
[2] **Wallhack:** An exploit that allows players to see through walls.

### Your Task

Build a binary classifier to automatically pre-filter reports and prioritize high-confidence suspicious accounts for manual review.

- **Metric:** F2
- **Formula:** `F2 = 5 · (P · R) / (4P + R)`
- **Target:** `is_cheater` (`0` = Legitimate, `1` = Cheater)
- **Features:** 33 total (32 predictive features + target)

Example features:

- `kill_death_ratio` (float): kills-to-deaths ratio
- `headshot_percentage` (float): percentage of kills that are headshots
- `win_rate` (float): overall win rate percentage
- `accuracy_score` (float): shot accuracy percentage
- `reaction_time_ms` (float): average reaction time in milliseconds

### Hints & Best Practices

- Handle missing values. Impute or drop them carefully.
- Handle class imbalance with resampling, class weights, or SMOTE.
- Use feature engineering to build interaction features.
- Use cross-validation for reliable performance.
- Try decision-threshold tuning.
- Use ensemble models for robustness.

## Task 2: Player Segment Classification

**Task category:** Multi-Class Classification

### Situation

Karena Thailand’s marketing team currently runs generic campaigns across all players, resulting in extremely low engagement rates of only 2–5%. The team spends approximately ฿60 million monthly on promotional campaigns, but most campaigns fall flat because they are not personalized. For example, the team shows competitive esports tournament announcements to casual mobile players who just want to relax, cosmetic item sales to players who have never made a purchase, and FiveN roleplay event invitations to hardcore PUBE competitive grinders. This misalignment wastes advertising budgets and annoys players with irrelevant content. The top 5% of players (whales) generate 60% of total revenue, but the company can’t identify the reliably using simple metrics like playtime alone. A high-playtime player could be a competitive grinder seeking glory, a social player enjoying friend time, or a whale collecting items. Each needs completely different messaging. Proper segmentation could increase campaign effectiveness by 3–4x, translating to millions of baht in additional revenue and improved player satisfaction.

### Your Task

Classify each player into one of four behavioral segments to enable targeted marketing and personalized experiences.

- **Metric:** F1
- **Formula:** `F1 = 2 · (P · R) / (P + R)`
- **Target:** `segment` (`0` = Casual, `1` = Grinder, `2` = Social, `3` = Whale)
- **Features:** 45 total (44 predictive features + target)
- **Illustration labels:** Casual Player, Competitive Grinder, Social Player, Whale (Big Spender)

Example features:

- `play_frequency` (float): average days per week played
- `avg_session_duration` (float): average session length in minutes
- `total_spending_thb` (float): lifetime spending
- `avg_monthly_spending` (float): average monthly spend
- `friend_count` (float): number of in-game friends

### Hints & Best Practices

- Use Task 1 tips for missing values and class imbalance.
- Use Stratified K-Fold to preserve class distribution.
- Try XGBoost, LightGBM, or CatBoost.
- Stack these models together for extra robustness.
- Engineer interaction and ratio features.

## Task 3: Player Monthly Spending Prediction

**Task category:** Regression

### Situation

Karena Thailand’s finance team struggles with revenue forecasting and consistently misses quarterly targets by 35–40%. This impacts budgeting, hiring decisions, and investor confidence. Without accurate player-level spending predictions, multiple departments operate blindly. The VIP support team of 20 people provides white-glove service but does not know which players deserve priority treatment. Last quarter, the team spent significant resources on 150 high-playtime players who collectively spent only ฿12,000, while a whale who spent ฿255,000 waited three days for a ticket response and then left for a competitor. The marketing team sends promotional discount codes randomly, often giving 50% off to whales who would have paid full price anyway. This results in ฿6 million+ in monthly revenue loss. Customer service wastes time on players unlikely to ever monetize, while high-value customers receive generic treatment. The company also needs to optimize limited-time event scheduling and new content releases based on when high-spending players are most active. Accurate spending predictions would enable proper resource allocation, targeted offers, revenue forecasting, and VIP program optimization across the organization.

### Your Task

Predict how much (in THB) each player will spend on in-game currency and items in the next 30 days.

- **Metric:** Normalized MAE
- **Formula:** `NMAE = 1 / (1 + Σ|y − ŷ| / Σy)`
- **Target:** `spending_30d` (continuous, THB)
- **Features:** 33 total (32 predictive features + target)

Example features:

- `friend_count` (float): number of in-game friends
- `event_participation_rate` (float): percentage of limited-time events joined
- `daily_login_streak` (float): current login streak
- `avg_session_length` (float): average session duration in minutes
- `historical_spending` (float): past spending

### Hints & Best Practices

- Zero-inflated data – try a two-stage model: (1) will spend? (2) how much?
- Use feature engineering and log transformation of the target.
- Try XGBoost, LightGBM, or CatBoost Regressors.
- Use ensembling or stacking for extra accuracy.
- Watch for common mistakes in target scaling.

## Task 4: Game Title Detection

**Task category:** Image Classification

### Situation

Karena Thailand operates five distinct games but can’t automatically identify which game is being played from screenshots. The customer support team receives 2,000 daily tickets with images and spends 10+ minutes per ticket figuring out which game to route to specialists. The “Play Multiple Games for Rewards” campaign can’t verify 50,000+ screenshot submissions, forcing manual checks of only 3% and leading to fraud. Players post 100,000+ gameplay images weekly across platforms, but the images can’t be auto-tagged for contests or moderation. Marketing can’t track which games are trending without manual review. An automated detection system would enable faster support, verified campaigns, scalable content moderation, and cross-game insights.

### Your Task

Build an image classifier that identifies which Karena game is being played from a single screenshot.

- **Metric:** F1 (Macro)
- **Formula:** `F1 = 2 · (P · R) / (P + R)`
- **Target:** `game_title` (5 classes)
- **Images:** Gameplay screenshots (144p, centered crop to 230 × 120)
- **Sources:** Live streams, submissions, forums, tickets
- **Variety:** Different maps, phases, weather, and UI states
- **Classes / submission IDs (per current TA addendum):** `0` = `7-11K`, `1` = `fiveN`, `2` = `freefried`, `3` = `pube`, `4` = `roblock`
- Pre-trained models are allowed for this task.

### Hints & Best Practices

- Review image classification basics and data augmentation.
- Watch for class imbalance in images.
- Try VGG16/VGG19, Vision Transformer (ViT), or Swin Transformer.
- Use cross-validation for CNNs. Ensemble methods help.
- You may need a GPU. Colab and Kaggle offer free T4 GPUs.
- Kaggle: approximately 30 GPU hours per week. Colab: dynamic, with no fixed limit.

## Task 5: Account Security Monitoring

**Task category:** Anomaly Detection

### Situation

Karena Thailand faces over 5,000 confirmed account thefts monthly. The actual number is likely 2x higher. Account theft is the #1 customer support issue and consumes 60% of team resources. Beyond theft, sophisticated account-selling rings destroy competitive integrity, bot farms operate 24/7 and cause game economy inflation, and players exploit VPNs for regional pricing (70% discounts), which costs millions of baht annually. The current rule-based security system has a 60% false positive rate. It flags legitimate travelers while missing smart attackers who wait 24 hours between actions. Simple thresholds flag players who legitimately improve while missing gradual botting patterns. The security team drowns in alerts while real threats slip through. The company needs an intelligent unsupervised anomaly detection system that learns normal behavior patterns and flags truly unusual activities without rigid rules or labeled data.

### Your Task

Build an unsupervised anomaly detection model to flag accounts with unusual behavior. Use techniques such as Isolation Forest or Autoencoders, without relying on labeled training data.

- **Metric:** F3
- **Formula:** `F3 = 10 · (P · R) / (9P + R)`
- **Target:** `is_anomaly` (`0` = Normal, `1` = Anomalous)
- **Features:** 33 total (32 predictive features + target)
- **Note:** No labeled training set is provided – `test.csv` only.

Example features:

- `login_count_1/2/3/4` (float): logins per time period
- `login_lat/lon_1/2/3/4` (float): login coordinates
- `location_changes_1/2/3/4` (float): significant location changes
- `device_count_1/2/3/4` (float): unique devices used

### Hints & Best Practices

- Try Isolation Forest, One-Class SVM, or Auto-Encoder.
- Unsupervised models can also be ensembled.
- Read about soft and hard voting techniques.
- Tune contamination or the threshold carefully.

## Submission Format

Fill your answers in `sample_submission.csv`. Include one prediction column for each task and fill every row for every task.

| `id`       | `task1` | `task2` | `task3` | `task4`     | `task5` |
| ---------- | ------: | ------: | ------: | ----------- | ------: |
| `ANS00001` |       0 |       2 |       0 | 2           |       1 |
| `ANS00002` |       1 |       3 |   10000 | 1           |       0 |
| `ANS00003` |       … |       … |       … | …           |       … |
| `ANS25889` |       … |       … |       … | …           |       … |

**Dataset size:** 25,889 rows × 6 columns.

## Evaluation Metrics

| Metric | Formula                        | Applied to     |
| ------ | ------------------------------ | -------------- |
| F1     | `F1 = 2 · (P · R) / (P + R)`   | Task 2, Task 4 |
| F2     | `F2 = 5 · (P · R) / (4P + R)`  | Task 1         |
| F3     | `F3 = 10 · (P · R) / (9P + R)` | Task 5         |
| NMAE   | `1 / (1 + Σ                    | y − ŷ          | / Σy)` | Task 3 |

## Kaggle Score Calculation

Each task is evaluated independently with its own metric. The task scores are combined into one weighted score that ranks your team on the Kaggle leaderboard.

Let `S1` through `S5` be the scores for Tasks 1 through 5. Let `w1` through `w5` be their weights.

`Weighted Score = (w₁S₁ + w₂S₂ + w₃S₃ + w₄S₄ + w₅S₅) / Σwᵢ`

**Score range:** `[0, 1]`

## Baseline Model

The first baseline uses only the models covered in class.

**Ground Baseline**  
Weighted Score (Public) = 0.68

Models shown: Logistic Regression, Random Forest, CNN, Isolation Forest.

The challenge baseline selects a model and uses techniques from the hints in this instruction.

**Challenge Baseline**  
Weighted Score (Public) = 0.75

To go beyond the baseline, try experimenting with cutting-edge models and techniques. We encourage you and your team to explore and experiment with new ideas!
