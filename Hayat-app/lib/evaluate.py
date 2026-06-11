from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

categories = ['Fire', 'Traffic', 'Municipality', 'Unclassified']

# التصنيف الصحيح
y_true = ['Fire'] * 36 + ['Traffic'] * 36 + ['Municipality'] * 68

# تصنيف Gemini
y_pred = ['Fire'] * 36 + ['Traffic'] * 36 + ['Municipality'] * 68

# الـ 2 اللي ما صنف Gemini
y_pred[2] = 'Unclassified'
y_pred[105] = 'Unclassified'

cm = confusion_matrix(y_true, y_pred, labels=categories)
cm_df = pd.DataFrame(cm, index=categories, columns=categories)
report = classification_report(y_true, y_pred, labels=categories, zero_division=0)
acc = accuracy_score(y_true, y_pred)

print("--- Confusion Matrix ---")
print(cm_df)
print("\n--- Classification Report ---")
print(report)
print(f"\nOverall Accuracy: {acc * 100:.2f}%")

# ===== الرسمة =====
plt.figure(figsize=(8,6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Hayat System')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()