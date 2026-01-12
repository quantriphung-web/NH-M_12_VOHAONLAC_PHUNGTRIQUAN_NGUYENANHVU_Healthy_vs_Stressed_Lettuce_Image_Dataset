
#Phùng Trí Quân

#CNN

# 6. EVALUATE
val_loss, val_acc = model.evaluate(val_gen, verbose=0)
print(f" Validation Accuracy: {val_acc:.4f}")
print(f" Validation Loss    : {val_loss:.4f}")

# 7. PLOT ACCURACY
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Accuracy theo Epoch")
plt.legend()
plt.grid(True)
plt.show()


#NAIVE BAYES
y_pred = nb_model.predict(X_val)
nb_acc = accuracy_score(y_val, y_pred)

print("\n Naive Bayes Accuracy:", nb_acc)
print("\n Báo cáo phân loại Naive Bayes:")
print(classification_report(y_val, y_pred, target_names=class_names))

#SO SÁNH CNN vs NAIVE BAYES
print("\n SO SÁNH MÔ HÌNH")
print(f"CNN Accuracy        : {val_acc:.4f}")
print(f"Naive Bayes Accuracy: {nb_acc:.4f}")





#Nguyễn Anh Vũ

#MobileNetV2

acc = accuracy_score(y_true, y_pred)
print(f"Accuracy trên tập validation: {acc:.4f}\n")

print("Classification Report:")
print(classification_report(
    y_true, y_pred,
    target_names=val_generator.class_indices.keys()
))

#Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=val_generator.class_indices.keys(),
            yticklabels=val_generator.class_indices.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()



#Võ Hoàn Lạc
#RESNET50

#Accuracy trên tập test
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

#Classification Report + Confusion Matrix
y_pred = (model.predict(X_test) > 0.5).astype(int)

print(classification_report(y_test, y_pred, target_names=class_names))
print(confusion_matrix(y_test, y_pred))

#Biểu đồ Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.legend(["Train","Validation"])
plt.show()


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

results = []





#Đánh giá chung tất cả các mô hình
# 1. CNN THUẦN 
y_true_cnn = val_gen.classes
y_pred_cnn = np.argmax(model_cnn.predict(val_gen), axis=1)

results.append({
    "Model": "CNN thuần",
    "Accuracy": accuracy_score(y_true_cnn, y_pred_cnn),
    "Precision": precision_score(y_true_cnn, y_pred_cnn, average="weighted"),
    "Recall": recall_score(y_true_cnn, y_pred_cnn, average="weighted"),
    "F1-score": f1_score(y_true_cnn, y_pred_cnn, average="weighted")
})


# 2. NAIVE BAYES
y_pred_nb = nb_model.predict(X_val)

results.append({
    "Model": "Naive Bayes",
    "Accuracy": accuracy_score(y_val, y_pred_nb),
    "Precision": precision_score(y_val, y_pred_nb, average="weighted"),
    "Recall": recall_score(y_val, y_pred_nb, average="weighted"),
    "F1-score": f1_score(y_val, y_pred_nb, average="weighted")
})


# 3. MOBILENETV2 (BINARY)
y_true_mn = val_generator.classes
y_pred_mn = (model_mobilenet.predict(val_generator) > 0.5).astype(int).ravel()

results.append({
    "Model": "MobileNetV2",
    "Accuracy": accuracy_score(y_true_mn, y_pred_mn),
    "Precision": precision_score(y_true_mn, y_pred_mn),
    "Recall": recall_score(y_true_mn, y_pred_mn),
    "F1-score": f1_score(y_true_mn, y_pred_mn)
})


# 4. RESNET50 (BINARY)
y_pred_rn = (model_resnet.predict(X_test) > 0.5).astype(int)

results.append({
    "Model": "ResNet50",
    "Accuracy": accuracy_score(y_test, y_pred_rn),
    "Precision": precision_score(y_test, y_pred_rn),
    "Recall": recall_score(y_test, y_pred_rn),
    "F1-score": f1_score(y_test, y_pred_rn)
})

# 5. BẢNG SO SÁNH
df_results = pd.DataFrame(results)
print("\nBẢNG SO SÁNH 4 MÔ HÌNH:\n")
print(df_results)
