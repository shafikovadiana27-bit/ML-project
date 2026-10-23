import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from imblearn.under_sampling import RandomUnderSampler


data = pd.read_csv('dataset.csv')

# Удаляем неизвестные машины
data = data[data['make'].notna() & data['model'].notna() & data['gear'].notna() & data['fuel'].notna()]

# Приводим к одному регистру
for col in ['make', 'model', 'fuel', 'gear', 'offerType']:
    data[col] = data[col].str.lower()

data['mileage'] = data['mileage'].abs()
data['price'] = data['price'].abs()
data['hp'] = data['hp'].abs()
data['year'] = data['year'].abs()

# Заполняем offerType
data.loc[(data['offerType'].isna()) & (data['mileage'] <= 15) & (data['year'] >= 2024), 'offerType'] = 'new'
data.loc[(data['offerType'].isna()) & (data['mileage'] <= 100) & (data['year'] >= 2023), 'offerType'] = 'pre-registered'
data.loc[(data['offerType'].isna()) & (data['mileage'] > 100) & (data['mileage'] <= 3000) & (data['year'] >= 2023), 'offerType'] = 'demonstration'
data.loc[(data['offerType'].isna()) & (data['mileage'] > 3000) & (data['mileage'] <= 10000) & (data['year'] >= 2022), 'offerType'] = "employee's car"
data['offerType'] = data['offerType'].fillna('used')

for  col in ['mileage', 'year']:
    data[col] = data[col].fillna(data.groupby('offerType')[col].transform('median'))

for col in ['price', 'hp']:
    data[col] = data[col].fillna(data.groupby(['make', 'model', 'year'])[col].transform('median'))
    data[col] = data[col].fillna(data.groupby(['make', 'model'])[col].transform('median'))
    data[col] = data[col].fillna(data.groupby('make')[col].transform('median'))

# Смена типа данных и удаление полных дубликатов
for col in ['mileage', 'price', 'hp', 'year']:
    data[col] = data[col].round().astype(int)

data = data.drop_duplicates()

data = data[
    data['mileage'].between(0, 1000000)
    & data['price'].between(100, 500000)
    & data['hp'].between(20, 1500)
    & data['year'].between(1950, 2026)
]

# Лейблинг и нормализация
# One-Hot - создаются отдельные бинарные столбцы
dummies = pd.get_dummies(data[['fuel', 'gear', 'offerType']], columns=['fuel', 'gear', 'offerType'])
data = pd.concat([data, dummies], axis=1)

# Frequency Encoding - словарь с частотой встречаемого значения и создаём новый столбец
# в котором для каждого значения make записана его частота
freq_make = data['make'].value_counts(normalize=True)
data['make_freq'] = data['make'].map(freq_make)

freq_model = data['model'].value_counts(normalize=True)
data['model_freq'] = data['model'].map(freq_model)

# Использует медиану и межквартильный размах, устойчив к еденичным огромным выбросам
data['hp_norm'] = RobustScaler().fit_transform(data[['hp']])

data['carAge_norm'] = RobustScaler().fit_transform(2026 - data[['year']])

data['make_freq_norm'] = RobustScaler().fit_transform(data[['make_freq']])

data['model_freq_norm'] = RobustScaler().fit_transform(data[['model_freq']])

# log1p нормализует значения, сжимает большие значения сильнее, чем малые
# (обычно на рынке много дешёвых машин и мало дорогих)
data['mileage_norm'] = StandardScaler().fit_transform(np.sqrt(data[['mileage']]))

data['price_norm'] = StandardScaler().fit_transform(np.log1p(data[['price']]))
price_scaler = StandardScaler().fit(np.log1p(data[['price']]))

# # Диаграммы рассеивания
# for col in ['mileage', 'hp', 'price']:
#     plt.figure(figsize=(12, 7))
#     sns.stripplot(y=data[col], alpha=0.3, color='red', jitter=0.35, size=3)
#     plt.title(f'Плотность распределения точек для {col}', fontsize=12)
#     plt.ylabel(col, fontsize=10)
#     plt.yscale('log')
#     plt.grid(True, linestyle='--', alpha=0.5)
# plt.show()

# # Ящики с усами
# plt.figure(figsize=(12, 7))
# box_data = data['price']
# plt.boxplot(box_data, labels=['All prices'])
# plt.yscale('log')
# plt.title('Распределение цены')
# plt.ylabel('Price, €')
# plt.show()

# plt.figure(figsize=(12, 7))
# box_data = data['year']
# plt.boxplot(box_data, labels=['All ages'])
# plt.title('Распределение по годам владения')
# plt.ylabel('Age')
# plt.show()

# # Гистограммы
# fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# # Исходное распределение
# sns.histplot(data['price'], bins=50, kde=True, ax=axes[0])
# axes[0].set_title(f"Price (skew = {data['price'].skew():.2f})")
# axes[0].set_xlabel('Price, €')
# # После log1p
# sns.histplot(np.log1p(data['price']), bins=50, kde=True, ax=axes[1])
# axes[1].set_title(f"log1p(Price) (skew = {np.log1p(data['price']).skew():.2f})")
# axes[1].set_xlabel('log1p(Price)')
# plt.tight_layout()
# plt.show()

# fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# # Исходное распределение
# sns.histplot(data['mileage'], bins=50, kde=True, ax=axes[0])
# axes[0].set_title(f"Mileage (skew = {data['mileage'].skew():.2f})")
# axes[0].set_xlabel('Mileage, KM')
# # После sqrt
# sns.histplot(np.sqrt(data['mileage']), bins=50, kde=True, ax=axes[1])
# axes[1].set_title(f"sqrt(Mileage) (skew = {np.sqrt(data['mileage']).skew():.2f})")
# axes[1].set_xlabel('sqrt(Mileage)')
# plt.tight_layout()
# plt.show()

# numeric = ['price', 'mileage', 'hp', 'year']

# # Матрица корреляции
# corr_matrix = data[numeric].corr(method='spearman')

# plt.figure(figsize=(8, 6))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
# plt.title('Матрица корреляции после очистки выбросов')
# plt.show()

# # ЭМПИРИЧЕСКАЯ ФУНКЦИЯ РАСПРЕДЕЛЕНИЯ (ECDF)
# ecdf_rows = []
# fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# for axis, column in zip(axes.flat, numeric):
#     if column == 'price':
#         values = np.log1p(data[column].to_numpy())
#         display_name = 'Цена'
#         axis_label = 'Цена, €'
#         inverse_transform = np.expm1
#     elif column == 'mileage':
#         values = np.sqrt(data[column].to_numpy())
#         display_name = 'Пробег'
#         axis_label = 'Пробег, км'
#         inverse_transform = np.square
#     elif column == 'hp':
#         values = np.log1p(data[column].to_numpy())
#         display_name = 'Мощность'
#         axis_label = 'Мощность, л. с.'
#         inverse_transform = np.expm1
#     else:
#         values = data[column].to_numpy()
#         display_name = 'Год выпуска'
#         axis_label = 'Год выпуска'
#         inverse_transform = lambda x: x

#     ecdf_result = stats.ecdf(values)
#     axis.step(
#         ecdf_result.cdf.quantiles,
#         ecdf_result.cdf.probabilities,
#         where='post',
#         color='royalblue',
#         linewidth=2,
#     )

#     q10, q25, q50, q75, q90 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
#     axis.axhline(0.50, color='darkorange', linestyle='--', linewidth=1)
#     axis.axvline(q50, color='darkorange', linestyle='--', linewidth=1)
#     axis.scatter(q50, 0.50, color='darkorange', zorder=3)
#     axis.set_title(f'{display_name}: ECDF')
#     axis.set_xlabel(axis_label)
#     axis.set_ylabel('Накопленная доля')
#     axis.set_ylim(0, 1.02)
#     axis.grid(alpha=0.25)

#     # Расчёт выполняется на преобразованной шкале, но подписи показывают
#     # исходные единицы: евро, километры, лошадиные силы и годы.
#     if column != 'year':
#         axis.xaxis.set_major_formatter(FuncFormatter(
#             lambda tick, position, inverse=inverse_transform:
#             f'{inverse(tick):,.0f}'.replace(',', ' ')
#         ))

#     ecdf_rows.append({
#         'Признак': display_name,
#         'P10': inverse_transform(q10),
#         'Q1': inverse_transform(q25),
#         'Медиана': inverse_transform(q50),
#         'Q3': inverse_transform(q75),
#         'P90': inverse_transform(q90),
#     })

# plt.suptitle('2. Эмпирические функции распределения')
# plt.tight_layout()
# plt.show()


# # ПРОВЕРКА НОРМАЛЬНОСТИ Д'АГОСТИНО-ПИРСОНА
# normality_samples = {
#     'log1p(price)': np.log1p(data['price'].to_numpy()),
#     'sqrt(mileage)': np.sqrt(data['mileage'].to_numpy()),
#     'log1p(hp)': np.log1p(data['hp'].to_numpy()),
#     'year': data['year'].to_numpy(),
# }
# normality_rows = []

# for name, values in normality_samples.items():
#     result = stats.normaltest(values)
#     normality_rows.append({
#         'Выборка': name,
#         'K²': result.statistic,
#         'p-value': result.pvalue,
#         'Вывод': (
#             'нормальность отвергается'
#             if result.pvalue < 0.05
#             else 'нет оснований отвергать нормальность'
#         ),
#     })

# fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# stats.probplot(normality_samples['log1p(price)'], dist='norm', plot=axes[0])
# axes[0].set_title('Q-Q: log1p(price)')
# stats.probplot(normality_samples['sqrt(mileage)'], dist='norm', plot=axes[1])
# axes[1].set_title('Q-Q: sqrt(mileage)')
# plt.suptitle('3. Проверка формы распределений')
# plt.tight_layout()
# plt.show()


# # РОБАСТНЫЙ АНАЛИЗ ВЫБРОСОВ С ПОМОЩЬЮ MAD
# mad_samples = {
#     'Цена': np.log1p(data['price'].to_numpy()),
#     'Пробег': np.sqrt(data['mileage'].to_numpy()),
#     'Мощность': np.log1p(data['hp'].to_numpy()),
#     'Год выпуска': data['year'].to_numpy(),
# }
# mad_rows = []

# for name, values in mad_samples.items():
#     median = np.median(values)
#     mad = stats.median_abs_deviation(values, scale='normal')

#     if mad == 0:
#         robust_z = np.zeros_like(values, dtype=float)
#     else:
#         robust_z = np.abs(values - median) / mad

#     outlier_count = int(np.sum(robust_z > 3.5))
#     outlier_share = 100 * outlier_count / len(values)

#     mad_rows.append({
#         'Признак': name,
#         'Медиана преобразованных значений': median,
#         'MAD': mad,
#         'Количество выбросов': outlier_count,
#         'Доля выбросов, %': outlier_share,
#     })

# mad_table = pd.DataFrame(mad_rows)

# plt.figure(figsize=(10, 6))
# bars = plt.bar(
#     mad_table['Признак'],
#     mad_table['Доля выбросов, %'],
#     color=['royalblue', 'darkorange', 'mediumseagreen', 'mediumpurple'],
# )
# upper_limit = max(mad_table['Доля выбросов, %'].max() * 1.25, 1)
# plt.ylim(0, upper_limit)
# plt.xlabel('Числовой признак')
# plt.ylabel('Доля робастных выбросов, %')
# plt.title('4. Робастный анализ выбросов с помощью MAD')
# plt.grid(axis='y', linestyle='--', alpha=0.3)

# for bar, percent, count in zip(
#     bars,
#     mad_table['Доля выбросов, %'],
#     mad_table['Количество выбросов'],
# ):
#     plt.text(
#         bar.get_x() + bar.get_width() / 2,
#         bar.get_height() + upper_limit * 0.025,
#         f'{percent:.2f}%\n(n={count})',
#         ha='center',
#         va='bottom',
#     )

# plt.figtext(
#     0.5,
#     0.01,
#     'Критерий выброса: |робастная z-оценка| > 3.5; '
#     'центр и масштаб рассчитаны по медиане и MAD',
#     ha='center',
#     fontsize=9,
#     color='dimgray',
# )
# plt.tight_layout(rect=(0, 0.05, 1, 1))
# plt.show()


# # ПОДБОР ТЕОРЕТИЧЕСКОГО РАСПРЕДЕЛЕНИЯ ЦЕНЫ
# price_values = data['price'].to_numpy(dtype=float)
# distribution_candidates = {
#     'Логнормальное': stats.lognorm,
#     'Гамма': stats.gamma,
#     'Вейбулла': stats.weibull_min,
# }
# distribution_rows = []
# fitted_distributions = {}

# for name, distribution in distribution_candidates.items():
#     parameters = distribution.fit(price_values, floc=0)
#     log_likelihood = np.sum(distribution.logpdf(price_values, *parameters))
#     parameter_count = len(parameters)
#     aic = 2 * parameter_count - 2 * log_likelihood

#     distribution_rows.append({
#         'Распределение': name,
#         'AIC': aic,
#         'Число параметров': parameter_count,
#     })
#     fitted_distributions[name] = (distribution, parameters)

# distribution_table = pd.DataFrame(distribution_rows).sort_values('AIC')
# best_distribution = distribution_table.iloc[0]['Распределение']

# plot_max = np.quantile(price_values, 0.995)
# plot_values = price_values[price_values <= plot_max]
# x_values = np.linspace(plot_values.min(), plot_max, 500)

# plt.figure(figsize=(11, 6))
# sns.histplot(
#     plot_values,
#     bins=60,
#     stat='density',
#     color='lightgray',
#     edgecolor='white',
#     label='Наблюдаемые цены',
# )

# for name, (distribution, parameters) in fitted_distributions.items():
#     plt.plot(
#         x_values,
#         distribution.pdf(x_values, *parameters),
#         linewidth=2,
#         label=name,
#     )

# plt.xlabel('Цена, €')
# plt.ylabel('Плотность вероятности')
# plt.title('5. Подбор распределения цены по минимальному AIC')
# plt.legend()
# plt.tight_layout()
# plt.show()



#===============> ОБУЧЕНИЕ МОДЕЛИ <===============


#~~~~~~~~~~ НЕПРЕРЫВНАЯ ВЕЛИЧИНА В КЛАСС ~~~~~~~~~~


#--------------------- KNN ---------------------

knn_feature_cols = (
                    ['hp_norm', 'carAge_norm', 'mileage_norm', 'make_freq_norm', 'model_freq_norm']
)

test_3_class = [0, 0.35, 0.8, 1.0]
data['price_class'] = pd.qcut(data['price'], q=test_3_class, labels=['cheap', 'medium', 'expensive'])

# test_4_class = [0, 0.35, 0.67, 0.86, 1.0]
# data['price_class'] = pd.qcut(data['price'], q=test_4_class, labels=['cheap', 'cheap-medium','medium-expensive', 'expensive'])

X = data[knn_feature_cols].astype(float)
le = LabelEncoder()
y = le.fit_transform(data['price_class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41, stratify=y)

neighbors = [1, 7, 13, 15, 17, 19, 21]
knn = GridSearchCV(
    KNeighborsClassifier(weights='distance'),
    param_grid={'n_neighbors' : neighbors},
    scoring='f1_macro',
    cv=5,
    n_jobs=-1
)

# knn = KNeighborsClassifier(weights='distance', n_neighbors=13)

knn.fit(X_train, y_train)

plt.figure(figsize=(10, 6))
plt.plot(neighbors, knn.cv_results_['mean_test_score'], marker='o')
plt.axvline(knn.best_params_['n_neighbors'], color='darkorange', linestyle='--', label=f'лучшее k = {knn.best_params_['n_neighbors']}')
plt.xlabel('Число соседей, k')
plt.ylabel('MAE на кросс-валидации')
plt.title('Подбор k для KNN-классификации')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

knn = knn.best_estimator_

y_pred = knn.predict(X_test)

print(classification_report(y_test, y_pred, target_names=le.classes_))

target_names = list(le.classes_)
cm = confusion_matrix(y_test, y_pred, normalize='true')

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)

disp.plot(cmap=plt.cm.Blues, values_format='.2f')

plt.title('Матрица предсказания категории цены', fontsize=14, pad=15)
plt.xlabel('Предсказанный класс', fontsize=12)
plt.ylabel('Реальный класс', fontsize=12)
plt.grid(False)
plt.tight_layout()
plt.show()

print('--------------------------------------------------------------')


#--------------------- SVM ---------------------

svecm_feature_cols = (
                    ['hp_norm', 'carAge_norm', 'mileage_norm', 'make_freq_norm', 'model_freq_norm']
)

test_3_class = [0, 0.35, 0.8, 1.0]
data['price_class'] = pd.qcut(data['price'], q=test_3_class, labels=['cheap', 'medium', 'expensive'])

# test_4_class = [0, 0.35, 0.67, 0.86, 1.0]
# data['price_class'] = pd.qcut(data['price'], q=test_4_class, labels=['cheap', 'cheap-medium','medium-expensive', 'expensive'])

X = data[svecm_feature_cols].astype(float)
le = LabelEncoder()
y = le.fit_transform(data['price_class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41, stratify=y)

# svecm = GridSearchCV(
#     SVC(kernel='rbf', class_weight='balanced'),
#     param_grid={'C': [0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 0.5]},
#     cv=5,
#     n_jobs=-1,
#     scoring='f1_macro'
# )

svecm = SVC(kernel='rbf', C=15, gamma=1, class_weight='balanced')

svecm.fit(X_train, y_train)

# svecm = svecm.best_estimator_

y_pred = svecm.predict(X_test)

print(classification_report(y_test, y_pred, target_names=le.classes_))

target_names = list(le.classes_)
cm = confusion_matrix(y_test, y_pred, normalize='true')

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)

disp.plot(cmap=plt.cm.Blues, values_format='.2f')

plt.title('Матрица предсказания категории цены', fontsize=14, pad=15)
plt.xlabel('Предсказанный класс', fontsize=12)
plt.ylabel('Реальный класс', fontsize=12)
plt.grid(False)
plt.tight_layout()
plt.show()

print('--------------------------------------------------------------')


#----------------- RandomForest -----------------

randfor_feature_cols = (
                    ['hp_norm', 'carAge_norm', 'mileage_norm', 'make_freq_norm', 'model_freq_norm']
)

test_3_class = [0, 0.35, 0.8, 1.0]
data['price_class'] = pd.qcut(data['price'], q=test_3_class, labels=['cheap', 'medium', 'expensive'])

# test_4_class = [0, 0.35, 0.67, 0.86, 1.0]
# data['price_class'] = pd.qcut(data['price'], q=test_4_class, labels=['cheap', 'cheap-medium','medium-expensive', 'expensive'])

X = data[randfor_feature_cols].astype(float)
le = LabelEncoder()
y = le.fit_transform(data['price_class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41, stratify=y)

# randfor = GridSearchCV(
#     RandomForestClassifier(class_weight='balanced_subsample', max_features='sqrt'),
#     param_grid={'n_estimators': [200, 275, 300], 'max_depth': [15, 18, 20], 'min_samples_leaf': [2, 3, 5]},
#     cv=3,
#     n_jobs=3,
#     scoring='f1_macro'
# )

randfor = RandomForestClassifier(n_estimators=300, max_depth=18, min_samples_leaf=2, class_weight='balanced_subsample', max_features='sqrt')

randfor.fit(X_train, y_train)

# randfor = randfor.best_estimator_

y_pred = randfor.predict(X_test)

print(classification_report(y_test, y_pred, target_names=le.classes_))

target_names = list(le.classes_)
cm = confusion_matrix(y_test, y_pred, normalize='true')

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)

disp.plot(cmap=plt.cm.Blues, values_format='.2f')

plt.title('Матрица предсказания категории цены', fontsize=14, pad=15)
plt.xlabel('Предсказанный класс', fontsize=12)
plt.ylabel('Реальный класс', fontsize=12)
plt.grid(False)
plt.tight_layout()
plt.show()

print('--------------------------------------------------------------')


#~~~~~~~~~~~~~~~~ С ИСПРАВЛЕНИЯМИ ~~~~~~~~~~~~~~~~


#--------------------- KNN ---------------------

knn_feature_cols = (
                    ['hp_norm', 'carAge_norm', 'mileage_norm', 'make_freq_norm', 'model_freq_norm']
)

test_3_class = [0, 0.35, 0.8, 1.0]
data['price_class'] = pd.qcut(data['price'], q=test_3_class, labels=['cheap', 'medium', 'expensive'])

# test_4_class = [0, 0.35, 0.67, 0.86, 1.0]
# data['price_class'] = pd.qcut(data['price'], q=test_4_class, labels=['cheap', 'cheap-medium','medium-expensive', 'expensive'])

X = data[knn_feature_cols].astype(float)
le = LabelEncoder()
y = le.fit_transform(data['price_class'])

neighbors = [1, 7, 13, 15, 17, 19, 21]
knn = GridSearchCV(
    KNeighborsClassifier(weights='distance'),
    param_grid={'n_neighbors' : neighbors},
    scoring='f1_macro',
    cv=5,
    n_jobs=-1
)

# knn = KNeighborsClassifier(weights='distance', n_neighbors=13)

knn.fit(X, y)

plt.figure(figsize=(10, 6))
plt.plot(neighbors, knn.cv_results_['mean_test_score'], marker='o')
plt.axvline(knn.best_params_['n_neighbors'], color='darkorange', linestyle='--', label=f'лучшее k = {knn.best_params_['n_neighbors']}')
plt.xlabel('Число соседей, k')
plt.ylabel('MAE на кросс-валидации')
plt.title('Подбор k для KNN-классификации')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

knn = knn.best_estimator_


#--------------------- SVM ---------------------

svecm_feature_cols = (
                    ['hp_norm', 'carAge_norm', 'mileage_norm', 'make_freq_norm', 'model_freq_norm']
)

test_3_class = [0, 0.35, 0.8, 1.0]
data['price_class'] = pd.qcut(data['price'], q=test_3_class, labels=['cheap', 'medium', 'expensive'])

# test_4_class = [0, 0.35, 0.67, 0.86, 1.0]
# data['price_class'] = pd.qcut(data['price'], q=test_4_class, labels=['cheap', 'cheap-medium','medium-expensive', 'expensive'])

X = data[svecm_feature_cols].astype(float)
le = LabelEncoder()
y = le.fit_transform(data['price_class'])

# svecm = GridSearchCV(
#     SVC(kernel='rbf', class_weight='balanced'),
#     param_grid={'C': [0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 0.5]},
#     cv=5,
#     n_jobs=-1,
#     scoring='f1_macro'
# )

svecm = SVC(kernel='rbf', C=15, gamma=1, class_weight='balanced')

svecm.fit(X, y)

# svecm = svecm.best_estimator_


#----------------- RandomForest -----------------

randfor_feature_cols = (
                    ['hp_norm', 'carAge_norm', 'mileage_norm', 'make_freq_norm', 'model_freq_norm']
)

test_3_class = [0, 0.35, 0.8, 1.0]
data['price_class'] = pd.qcut(data['price'], q=test_3_class, labels=['cheap', 'medium', 'expensive'])

# test_4_class = [0, 0.35, 0.67, 0.86, 1.0]
# data['price_class'] = pd.qcut(data['price'], q=test_4_class, labels=['cheap', 'cheap-medium','medium-expensive', 'expensive'])

X = data[randfor_feature_cols].astype(float)
le = LabelEncoder()
y = le.fit_transform(data['price_class'])

# randfor = GridSearchCV(
#     RandomForestClassifier(class_weight='balanced_subsample', max_features='sqrt'),
#     param_grid={'n_estimators': [200, 275, 300], 'max_depth': [15, 18, 20], 'min_samples_leaf': [2, 3, 5]},
#     cv=3,
#     n_jobs=3,
#     scoring='f1_macro'
# )

randfor = RandomForestClassifier(n_estimators=300, max_depth=18, min_samples_leaf=2, class_weight='balanced_subsample', max_features='sqrt')

randfor.fit(X, y)

# randfor = randfor.best_estimator_