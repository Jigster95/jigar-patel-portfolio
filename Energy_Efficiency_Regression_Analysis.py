

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score


# Data reading

col_names = [
    'Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area',
    'Overall_Height', 'Orientation', 'Glazing_Area', 'Glazing_Area_Distribution',
    'Heating_Load', 'Cooling_Load'
]

data = pd.read_excel('ENB2012_data.xlsx', header=0)
data.columns = col_names

print(len(data))

print(data.head())


# EDA

print(f"\nMissing values:\n{data.isnull().sum()}")

print(f"\nDescriptive statistics:\n{data.describe().round(3)}")

print(f"\nHeating Load stats:\n{data['Heating_Load'].describe()}")
print(f"\nCooling Load stats:\n{data['Cooling_Load'].describe()}")

feature_cols = col_names[:8]

fig, axes = plt.subplots(2, 5, figsize=(18, 7))
for ax, col in zip(axes.flatten(), col_names):
    ax.hist(data[col], bins=20, color='steelblue', edgecolor='white')
    ax.set_title(col, fontsize=8)
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
plt.suptitle('Feature & Target Distributions', fontsize=13)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Pearson Correlation Matrix')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 4, figsize=(16, 6))
for ax, col in zip(axes.flatten(), feature_cols):
    ax.boxplot(data[col])
    ax.set_title(col, fontsize=8)
plt.suptitle('Box Plots — Outlier Check', fontsize=13)
plt.tight_layout()
plt.show()


# Data preprocessing and train/test split

X = data[feature_cols].values
y = data['Heating_Load'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Model training and evaluation

results = {}


# Linear Regression (baseline)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

results['Linear Regression'] = {
    'mse': mean_squared_error(y_test, y_pred_lr),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
    'r2': r2_score(y_test, y_pred_lr)
}

print(f"\nLinear Regression Results:")
print(f"MSE:  {results['Linear Regression']['mse']:.4f}")
print(f"RMSE: {results['Linear Regression']['rmse']:.4f}")
print(f"R2:   {results['Linear Regression']['r2']:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(y_test, y_pred_lr, alpha=0.6, color='steelblue')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0].set_xlabel('Actual Heating Load')
axes[0].set_ylabel('Predicted Heating Load')
axes[0].set_title('Linear Regression — Actual vs Predicted')
axes[1].scatter(y_pred_lr, y_test - y_pred_lr, alpha=0.6, color='coral')
axes[1].axhline(0, linestyle='--', color='black')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Linear Regression — Residuals')
plt.tight_layout()
plt.show()


# Polynomial Regression (degree=2)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

lr_poly = LinearRegression()
lr_poly.fit(X_train_poly, y_train)
y_pred_poly = lr_poly.predict(X_test_poly)

results['Polynomial Regression (degree=2)'] = {
    'mse': mean_squared_error(y_test, y_pred_poly),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_poly)),
    'r2': r2_score(y_test, y_pred_poly)
}

print(f"\nPolynomial Regression (degree=2) Results:")
print(f"MSE:  {results['Polynomial Regression (degree=2)']['mse']:.4f}")
print(f"RMSE: {results['Polynomial Regression (degree=2)']['rmse']:.4f}")
print(f"R2:   {results['Polynomial Regression (degree=2)']['r2']:.4f}")

degrees = [1, 2, 3, 4]
train_rmse, test_rmse = [], []
for d in degrees:
    p = PolynomialFeatures(degree=d, include_bias=False)
    Xtr = p.fit_transform(X_train_scaled)
    Xte = p.transform(X_test_scaled)
    m = LinearRegression().fit(Xtr, y_train)
    train_rmse.append(np.sqrt(mean_squared_error(y_train, m.predict(Xtr))))
    test_rmse.append(np.sqrt(mean_squared_error(y_test, m.predict(Xte))))

plt.figure(figsize=(7, 4))
plt.plot(degrees, train_rmse, 'o-', label='Train RMSE')
plt.plot(degrees, test_rmse, 's--', label='Test RMSE')
plt.xlabel('Polynomial Degree')
plt.ylabel('RMSE')
plt.title('Bias-Variance Trade-off — Polynomial Degree')
plt.legend()
plt.tight_layout()
plt.show()


# Ridge Regression with hyperparameter tuning

kf = KFold(n_splits=5, shuffle=True, random_state=42)
ridge = Ridge()
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100, 1000]}
grid_ridge = GridSearchCV(ridge, param_grid, scoring='neg_root_mean_squared_error', cv=kf, n_jobs=1)
grid_ridge.fit(X_train_poly, y_train)

print(f"\nRidge best alpha: {grid_ridge.best_params_['alpha']}")
print(f"Ridge best CV RMSE: {-grid_ridge.best_score_:.4f}")

y_pred_ridge = grid_ridge.predict(X_test_poly)

results['Ridge Regression'] = {
    'mse': mean_squared_error(y_test, y_pred_ridge),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
    'r2': r2_score(y_test, y_pred_ridge)
}

print(f"\nRidge Regression Results:")
print(f"MSE:  {results['Ridge Regression']['mse']:.4f}")
print(f"RMSE: {results['Ridge Regression']['rmse']:.4f}")
print(f"R2:   {results['Ridge Regression']['r2']:.4f}")

alphas = param_grid['alpha']
mean_scores = -grid_ridge.cv_results_['mean_test_score']
plt.figure(figsize=(7, 4))
plt.semilogx(alphas, mean_scores, 'o-', color='darkorange')
plt.axvline(grid_ridge.best_params_['alpha'], linestyle='--', color='red',
            label=f"Best alpha={grid_ridge.best_params_['alpha']}")
plt.xlabel('Alpha (log scale)')
plt.ylabel('CV RMSE')
plt.title('Ridge — Alpha Tuning')
plt.legend()
plt.tight_layout()
plt.show()



# Model comparison

print("\nModel Comparison:")
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"  MSE:  {metrics['mse']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R2:   {metrics['r2']:.4f}")

results_df = pd.DataFrame(results).T.reset_index()
results_df.columns = ['Model', 'MSE', 'RMSE', 'R2']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].barh(results_df['Model'], results_df['RMSE'])
axes[0].set_xlabel('RMSE (lower is better)')
axes[0].set_title('Test RMSE — Model Comparison')
axes[0].invert_yaxis()
axes[1].barh(results_df['Model'], results_df['R2'])
axes[1].set_xlabel('R2 (higher is better)')
axes[1].set_title('Test R2 — Model Comparison')
axes[1].invert_yaxis()
plt.tight_layout()
plt.show()

preds = {
    'Linear Regression': y_pred_lr,
    'Poly Regression (d=2)': y_pred_poly,
    'Ridge Regression': y_pred_ridge,
}

fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
for ax, (name, yp) in zip(axes, preds.items()):
    ax.scatter(y_test, yp, alpha=0.5, s=20)
    lims = [y_test.min() - 1, y_test.max() + 1]
    ax.plot(lims, lims, 'r--', lw=1)
    ax.set_xlim(lims)
    ax.set_title(name, fontsize=9)
    ax.set_xlabel('Actual')
axes[0].set_ylabel('Predicted Heating Load')
plt.suptitle('Actual vs Predicted — All Models')
plt.tight_layout()
plt.show()
