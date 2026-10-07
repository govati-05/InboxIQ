# InboxIQ — Viva Questions & Answers

Simple, student-friendly answers for your ML Lab viva.

---

**1. Why did you choose this project?**
Email overload is a real everyday problem, and it lets me demonstrate a complete ML pipeline — preprocessing, TF-IDF, multiple classifiers, evaluation, and clustering — in one useful application.

**2. Why did you use the Gmail API instead of a fake/sample inbox?**
To make the project a genuinely working tool, not just a toy demo. I also built a Demo Mode using synthetic data so the project can be presented without needing a live Gmail connection.

**3. Why OAuth 2.0 instead of asking for a password?**
OAuth lets the user grant limited, revocable access without ever sharing their Gmail password with the app. It's the standard, secure way to access Google APIs.

**4. Why did you request only the `gmail.readonly` scope?**
Because InboxIQ only needs to read emails to classify and display them — it never sends, deletes, or modifies anything. Requesting the minimum necessary permission is a core security best practice (principle of least privilege).

**5. What is TF-IDF?**
Term Frequency–Inverse Document Frequency. It converts text into numbers by scoring each word higher if it appears often in one email but rarely across all emails — so it highlights words that are actually distinctive.

**6. Why did you use TF-IDF instead of just word counts?**
Plain word counts overweight common words (like "the", "please") that appear everywhere and don't help distinguish categories. TF-IDF automatically down-weights those.

**7. What is Naive Bayes?**
A probabilistic classifier based on Bayes' theorem, which assumes features (words) are independent of each other given the class. It's fast and works surprisingly well for text classification.

**8. Why did you include Naive Bayes?**
It's a classic, lightweight baseline for text classification and is very fast to train, which makes it a good point of comparison against more complex models.

**9. What is Logistic Regression?**
A linear model that predicts the probability of each class using a weighted sum of input features passed through a sigmoid/softmax function.

**10. What is a Decision Tree?**
A model that splits data step-by-step using if/else conditions on feature values, forming a tree structure, until it reaches a predicted class at the leaves.

**11. What is Random Forest?**
An ensemble of many decision trees, each trained on a random subset of data/features; their predictions are combined (majority vote) to reduce overfitting and improve accuracy compared to a single tree.

**12. Why did you compare multiple algorithms instead of picking just one?**
Different algorithms have different strengths. Comparing them with real metrics lets the code automatically pick the actually best-performing one for each task, instead of guessing.

**13. What is a train-test split?**
Splitting the dataset into a training portion (used to fit the model) and a separate testing portion (used only to evaluate it), so we measure performance on data the model hasn't already memorized.

**14. Why did you use a *stratified* train-test split?**
Stratification keeps the same class proportions in both the train and test sets, which is important with multiple categories/priorities so no split accidentally under-represents a class.

**15. What is overfitting?**
When a model learns the training data too specifically (including its noise) and performs well on training data but poorly on new, unseen data.

**16. How did you check InboxIQ isn't overfitting?**
By evaluating strictly on the held-out test split (data the model never saw during training) and reporting those real metrics, not training-set metrics.

**17. What is precision?**
Of all the emails the model predicted as a given class, the fraction that were actually that class. High precision = few false alarms.

**18. What is recall?**
Of all the emails that truly belong to a given class, the fraction the model correctly found. High recall = few missed cases.

**19. What is F1-score?**
The harmonic mean of precision and recall — a single balanced number that penalizes models that are good at only one of the two.

**20. What is a confusion matrix?**
A table showing actual classes (rows) vs predicted classes (columns), so you can see exactly which categories the model confuses with each other.

**21. What is PCA?**
Principal Component Analysis — a technique that reduces many numeric dimensions (like hundreds of TF-IDF features) down to a smaller number (e.g. 2) while keeping as much of the original variation/information as possible.

**22. Why did you use PCA in this project?**
Purely to visualize high-dimensional TF-IDF vectors on a 2D scatter plot — it's impossible to plot 500+ dimensions directly, so PCA compresses them to 2 axes we can actually see.

**23. What is K-Means?**
An unsupervised clustering algorithm that groups data points into K clusters by repeatedly assigning each point to its nearest cluster center and then recomputing the centers.

**24. What's the difference between classification and clustering?**
Classification is *supervised* — the model learns from data that already has correct labels (category/priority). Clustering is *unsupervised* — K-Means groups emails using only their text similarity, with no labels at all.

**25. How does Gmail authentication work in InboxIQ?**
On first run, InboxIQ opens a Google sign-in page via OAuth. After you approve read-only access, Google returns a credential token, which InboxIQ saves to `token.json` so you don't have to log in every single time.

**26. What is `credentials.json`?**
The OAuth client configuration file you download from Google Cloud Console — it identifies InboxIQ as an authorized application to Google. It contains no personal data by itself.

**27. What is `token.json`?**
The file InboxIQ creates *after* you approve access — it stores your personal access/refresh token for your Gmail account. Unlike `credentials.json`, this file is specific to your authorization.

**28. Why should `credentials.json` and `token.json` never be uploaded to GitHub?**
`token.json` grants direct access to your real Gmail data, and `credentials.json` identifies your OAuth app — both are secrets. If leaked publicly, someone else could impersonate the app or your access. Both are listed in `.gitignore` for this reason.

**29. How does a new Gmail email get classified end-to-end?**
Its subject/body are cleaned and combined → converted to a TF-IDF vector → passed to the trained category model → the predicted category (plus text/metadata features) is passed to the trained priority model → a 0–100 priority score is computed → everything is saved to SQLite and shown in the UI.

**30. How is priority predicted?**
A separate ML model (chosen automatically by comparing Naive Bayes, Logistic Regression, Decision Tree, and Random Forest) is trained on TF-IDF text features plus engineered features (subject/body length, urgency/deadline/security keyword counts) plus the predicted category, to classify each email as HIGH, MEDIUM, or LOW.

**31. How is the priority score (0–100) calculated?**
Two clearly separate parts: (1) an **ML-derived base score** computed from the priority model's class probabilities weighted against fixed anchor points (LOW=15, MEDIUM=50, HIGH=85), and (2) a small, capped **rule-based adjustment** (max ±10) based on how many urgency/deadline/security keywords were detected. The app shows both numbers separately so it never pretends a rule-based number is a pure ML prediction.

**32. Why is category classification accuracy so high (often ~100%) in your results?**
Because the synthetic training templates use highly distinctive vocabulary per category (e.g. "exam"/"semester" for Education vs "discount"/"sale" for Promotions), which makes the categories very easy to separate linearly. On messier real-world email text, accuracy would likely be somewhat lower — this is explained transparently in the README's Limitations section.

**33. What are the main limitations of this project?**
It's trained on synthetic (not real) data, so real Gmail accuracy may differ; it doesn't process attachments; the priority score blends ML output with a small rule-based nudge; and it currently doesn't personalize to individual user feedback.

**34. What are some future enhancements you'd consider?**
Gmail label integration, spam/duplicate detection, automatic summarization, personalized priority learning from user corrections, multilingual support, attachment analysis, and cloud deployment.

**35. Why did you cache emails in SQLite instead of always calling the Gmail API?**
To avoid unnecessary repeated API calls and re-classification, which is slower and wastes API quota. SQLite lets InboxIQ remember which `message_id`s it has already fetched and skip them on the next sync.
