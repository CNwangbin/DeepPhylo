import matplotlib.pyplot as plt
import numpy as np
def plot_training(train_losses, val_losses, metrics, title='The twin prediction result: train and test Loss/metrics'):
    def plot_metric(name, metric_values):
        max_metric = max(metric_values)
        plt.plot(metric_values, label=f"{name}")
        # show the digits of max validation aupr in the figure
        plt.plot(metric_values.index(max_metric), max_metric, 'ro')
        # show the digits of max validation aupr in the figure
        plt.annotate(f'{max_metric:.4f}', xy=(metric_values.index(max_metric), max_metric), xytext=(metric_values.index(max_metric), max_metric))
      
    # plot the training and validation loss
    plt.figure(figsize=(10, 10))
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plot_metric('acc', metrics['acc'])
    plot_metric('mcc', metrics['mcc'])
    plot_metric('roc_auc', metrics['roc_auc'])
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss/AUC')
    plt.legend(frameon=False)

def plot_pr_curve(precision, recall, title='Precision-Recall Curve'):
    # 绘制 Precision-Recall 曲线
    plt.figure()
    plt.step(recall, precision, color='black', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    # 绘制y=x参考线
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    # 计算最接近 y=x 的点
    diff = np.abs(np.array(precision) - np.array(recall))
    min_diff_idx = np.argmin(diff)
    intersect_x = recall[min_diff_idx]
    intersect_y = precision[min_diff_idx]
    
    # 绘制最接近 y=x 的点
    plt.scatter(intersect_x, intersect_y, color='r')
    plt.text(intersect_x, intersect_y, f'({intersect_x:.4f}, {intersect_y:.4f})', 
             verticalalignment='bottom', horizontalalignment='right')
    plt.title(title)
    #plt.legend()
    return intersect_x, intersect_y

def plot_ss_curve(sensitivity, specificity, title='Sensitivity-Specificity Curve'):
    # 绘制 Sensitivity-Specificity 曲线
    plt.figure()
    plt.step(specificity, sensitivity, color='black', alpha=0.2, where='post')
    plt.fill_between(specificity, sensitivity, step='post', alpha=0.2, color='lightskyblue')
    plt.xlabel('Specificity')
    plt.ylabel('Sensitivity')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    # 绘制y=x参考线, 以及与曲线相交的点
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    # 计算最接近 y=x 的点
    diff = np.abs(np.array(sensitivity) - np.array(specificity))
    min_diff_idx = np.argmin(diff)
    intersect_x = specificity[min_diff_idx]
    intersect_y = sensitivity[min_diff_idx]
    
    # 绘制最接近 y=x 的点
    plt.scatter(intersect_x, intersect_y, color='r', label='Intersection')
    plt.text(intersect_x, intersect_y, f'({intersect_x:.4f}, {intersect_y:.4f})', 
             verticalalignment='bottom', horizontalalignment='right')
    plt.title(title)
    return intersect_x, intersect_y

def plot_age(train_losses, val_losses, val_r2s, title='The simple MLP baseline age prediction result: train and test Loss/R2'):
    # Plot the training and validation
    max_r2 = max(val_r2s)
    # plot the training and validation loss
    plt.figure()
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.plot(val_r2s, label='Validation R2')
    # show the digits of max validation aupr in the figure
    plt.plot(val_r2s.index(max_r2), max_r2, 'ro')
    # show the digits of max validation aupr in the figure
    plt.annotate(f'{max_r2:.4f}', xy=(val_r2s.index(max_r2), max_r2), xytext=(val_r2s.index(max_r2), max_r2))
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss/R2')
    plt.legend(frameon=False)
    plt.show()