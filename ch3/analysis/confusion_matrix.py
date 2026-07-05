from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 假设的真实标签和预测标签，用于示例
true_levels = [3, 2, 2, 3, 2, 2, 1, 0, 1, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 3, 1, 0, 3, 2, 3]
pred_levels = [3, 2, 2, 3, 2, 2, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 3, 0, 0, 3, 0, 0]

# 计算混淆矩阵
cm = confusion_matrix(true_levels, pred_levels)

# 绘制混淆矩阵
fig, ax = plt.subplots(figsize=(8, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title('Confusion Matrix')
plt.savefig("matrix.png")
plt.show()