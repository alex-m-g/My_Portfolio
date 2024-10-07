import matplotlib.pyplot as plt
import seaborn as sns

# Generate plots
sns.countplot(x='Class', data=df)
plt.savefig('./output/class_distribution.png')
