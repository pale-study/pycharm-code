import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 生成示例数据（如果还没有数据文件）
def generate_sample_data():
    np.random.seed(42)
    n_records = 1000
    student_ids = [f"STU_{i:04d}" for i in range(1, 11)]  # 10个学生

    dates = pd.date_range('2023-09-01', '2023-12-10', freq='D')[:n_records]

    data = {
        'student_id': np.random.choice(student_ids, n_records),
        'date': np.random.choice(dates, n_records),
        'food': np.round(np.clip(np.random.normal(30, 10, n_records), 5, 80)),
        'study_materials': np.round(np.clip(np.random.exponential(8, n_records), 0, 50)),
        'transport': np.random.choice([0, 0, 0, 5, 5, 10, 15, 20], n_records,
                                      p=[0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]),
        'entertainment': np.round(np.clip(np.random.normal(25, 12, n_records), 0, 80)),
        'daily_supplies': np.random.choice([0, 0, 5, 10, 20, 30], n_records, p=[0.6, 0.2, 0.1, 0.05, 0.03, 0.02])
    }

    df = pd.DataFrame(data)
    df['total'] = df['food'] + df['study_materials'] + df['transport'] + df['entertainment'] + df['daily_supplies']
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['student_id', 'date'])

    return df


# 加载或生成数据
try:
    df = pd.read_csv('student_expenses_100000.csv')
    df['date'] = pd.to_datetime(df['date'])
    print("从文件加载数据成功")
except:
    print("数据文件不存在，生成示例数据...")
    df = generate_sample_data()
    print("示例数据生成完成")

print(f"数据形状: {df.shape}")
print(f"学生数量: {df['student_id'].nunique()}")
print(f"时间范围: {df['date'].min()} 到 {df['date'].max()}")


# 1. 所有学生日常开销分布饼图
def plot_expense_distribution_pie():
    # 计算各类别总开销
    categories = ['food', 'study_materials', 'transport', 'entertainment', 'daily_supplies']
    category_totals = df[categories].sum()
    category_percentages = (category_totals / category_totals.sum() * 100).round(1)

    # 中文标签
    category_labels = ['餐饮', '学习资料', '交通', '娱乐', '日用品']

    # 创建饼图
    plt.figure(figsize=(12, 8))

    # 设置颜色
    colors = ['#FF9999', '#66B3FF', '#99FF99', '#FFCC99', '#C9A0DC']

    # 绘制饼图
    wedges, texts, autotexts = plt.pie(
        category_totals,
        labels=category_labels,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        shadow=True
    )

    # 美化文字
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)

    plt.title('所有学生日常开销分布', fontsize=16, fontweight='bold', pad=20)
    plt.axis('equal')  # 确保饼图是圆形

    # 添加图例
    legend_labels = [f'{label}: {value}元 ({percentage}%)'
                     for label, value, percentage in zip(category_labels, category_totals, category_percentages)]
    plt.legend(wedges, legend_labels, title="开销类别", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    plt.tight_layout()
    plt.show()

    # 打印统计信息
    print("\n=== 所有学生开销统计 ===")
    for i, (label, total, percentage) in enumerate(zip(category_labels, category_totals, category_percentages)):
        print(f"{label}: {total:,}元 ({percentage}%)")


# 2. 某个学生各种开销的变化折线图
def plot_student_expense_trend(student_id=None):
    if student_id is None:
        # 随机选择一个学生
        student_id = np.random.choice(df['student_id'].unique())
        print(f"随机选择学生: {student_id}")

    # 筛选该学生的数据
    student_data = df[df['student_id'] == student_id].copy()

    if len(student_data) == 0:
        print(f"找不到学生 {student_id} 的数据")
        return

    # 按日期排序
    student_data = student_data.sort_values('date')

    # 创建折线图
    plt.figure(figsize=(15, 10))

    # 绘制每条开销类别的折线
    categories = ['food', 'study_materials', 'transport', 'entertainment', 'daily_supplies']
    category_labels = ['餐饮', '学习资料', '交通', '娱乐', '日用品']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']

    for i, (category, label, color) in enumerate(zip(categories, category_labels, colors)):
        plt.plot(student_data['date'], student_data[category],
                 label=label, color=color, linewidth=2.5, marker='o', markersize=4)

    # 绘制总开销折线
    plt.plot(student_data['date'], student_data['total'],
             label='总开销', color='#6A0DAD', linewidth=3, linestyle='--', marker='s', markersize=5)

    # 设置图表标题和标签
    plt.title(f'学生 {student_id} 各种开销变化趋势', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('开销金额 (元)', fontsize=12)

    # 设置x轴日期格式
    plt.gcf().autofmt_xdate()

    # 添加网格
    plt.grid(True, alpha=0.3)

    # 添加图例
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # 添加平均值线
    for i, category in enumerate(categories):
        avg_value = student_data[category].mean()
        plt.axhline(y=avg_value, color=colors[i], linestyle=':', alpha=0.7)
        plt.text(student_data['date'].iloc[-1] + pd.Timedelta(days=1), avg_value,
                 f'{avg_value:.1f}', va='center', color=colors[i], fontweight='bold')

    plt.tight_layout()
    plt.show()

    # 打印该学生的统计信息
    print(f"\n=== 学生 {student_id} 开销统计 ===")
    print(f"记录天数: {len(student_data)}")
    print(f"总开销: {student_data['total'].sum():.2f}元")
    print(f"日均开销: {student_data['total'].mean():.2f}元")
    print("\n各类别平均开销:")
    for category, label in zip(categories, category_labels):
        avg = student_data[category].mean()
        percentage = (avg / student_data['total'].mean() * 100) if student_data['total'].mean() > 0 else 0
        print(f"  {label}: {avg:.2f}元 ({percentage:.1f}%)")


# 3. 额外分析：开销的时间分布
def plot_expense_time_analysis():
    # 按月份分析
    df['month'] = df['date'].dt.month
    monthly_expenses = df.groupby('month')['total'].agg(['mean', 'sum', 'count'])

    # 按星期分析
    df['weekday'] = df['date'].dt.day_name()
    weekday_expenses = df.groupby('weekday')['total'].mean()
    # weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # weekday_expenses = df.groupby('weekday')['total'].mean().reindex(weekday_order)
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 月度开销柱状图
    months = ['9月', '10月', '11月', '12月']
    ax1.bar(months, monthly_expenses['mean'], color=['#FF9999', '#66B3FF', '#99FF99', '#FFCC99', '#FECA57'])
    ax1.set_title('各月平均每日开销', fontsize=14, fontweight='bold')
    ax1.set_ylabel('平均开销 (元)')

    # 添加数值标签
    for i, v in enumerate(monthly_expenses['mean']):
        ax1.text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

    # 星期开销柱状图
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_labels = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    weekday_means = [weekday_expenses.get(day, 0) for day in weekdays]

    ax2.bar(weekday_labels, weekday_means,
            color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF'])
    ax2.set_title('各星期平均开销', fontsize=14, fontweight='bold')
    ax2.set_ylabel('平均开销 (元)')

    # 添加数值标签
    for i, v in enumerate(weekday_means):
        ax2.text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()


# 执行分析
print("\n" + "=" * 50)
print("开始分析学生开销数据")
print("=" * 50)

# 绘制所有学生开销分布饼图
plot_expense_distribution_pie()

print("\n" + "=" * 50)

# 绘制某个学生开销变化折线图
# 可以选择特定学生或随机选择
plot_student_expense_trend()  # 随机选择学生
# plot_student_expense_trend("STU_0001")  # 指定学生ID

print("\n" + "=" * 50)

# 绘制时间分析图
plot_expense_time_analysis()

print("\n分析完成！")