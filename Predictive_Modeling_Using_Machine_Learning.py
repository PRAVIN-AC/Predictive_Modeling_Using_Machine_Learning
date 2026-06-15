import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. DATA FETCHING & CLEANING (From Task 1)
# ==========================================
print("📥 Fetching and cleaning dataset...")
url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"
df = pd.read_csv(url)

# Clean missing values using median/mode strategies
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(include=[object]).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# ==========================================
# 2. FEATURE ENGINEERING & PREPARATION
# ==========================================
print("⚙️ Preparing features for Predictive Modeling...")

# Define features (X) and Target (y)
# We will use structural body measurements to predict the Species
features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
X = df[features]
y = df['species']

# Split the dataset into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# 3. TRAINING THE ML MODEL (Random Forest)
# ==========================================
print("🤖 Training the Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Training Complete!")
print(f"🎯 Test Accuracy: {accuracy * 100:.2f}%")

# ==========================================
# 4. VISUALIZE PERFORMANCE (Confusion Matrix)
# ==========================================
print("\n📊 Generating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
labels = np.unique(y)

plt.figure(figsize=(7, 5))
sns.set_theme(style="white")
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=False)

plt.title('Confusion Matrix - Penguin Species Predictor', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Predicted Species', fontsize=12, labelpad=10)
plt.ylabel('Actual Species', fontsize=12, labelpad=10)
plt.tight_layout()

# Save the metric plot
output_chart = 'model_evaluation_matrix.png'
plt.savefig(output_chart, dpi=300)
print(f"💾 Evaluation plot saved as '{output_chart}'")
plt.show()

# ==========================================
# 5. TECHNICAL PORTFOLIO REPORT
# ==========================================
print("\n📝 ==========================================")
print("         MODEL PERFORMANCE REPORT             ")
print("==============================================")
print(classification_report(y_test, y_pred))
print("==============================================")