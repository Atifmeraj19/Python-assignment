import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

test = pd.read_csv('test.csv', encoding='ISO-8859-1')
test.head()
train = pd.read_csv('train.csv', encoding='ISO-8859-1')
train.head()
ideal = pd.read_csv('ideal.csv', encoding='ISO-8859-1')
ideal.head()

#Extract x and y values from datasets
test_x = test.iloc[:, 0]
test_y = test.iloc[:, 1]

train_x = train.iloc[:, 0]
train_y = train.iloc[:, 1:]

# Step 1: Iterate through ideal functions and calculate fit to training data
errors = []
for col in ideal.columns:
    ideal_function = ideal[col]
    error_sum = 0
    for i in range(1, 5):
        # Interpolate ideal function values at training data x-values
        training_data = train_y[f"y{i}"]
        interpolated_values = np.interp(train_x, ideal.iloc[:, 0], ideal_function)
        # Calculate sum of squared deviations (Least - Square criterion )
        error_sum += np.sum((training_data - interpolated_values) ** 2)
    errors.append(error_sum)

# Step 2: Select four ideal functions with lowest error on training data
best_fit_indices = np.argsort(errors)[:4]
best_fit_functions = [ideal.iloc[:, index] for index in best_fit_indices]

# Step 3: Test selected functions on test dataset
test_errors = []
for best_fit_function in best_fit_functions:
    interpolated_values = np.interp(test_x, ideal.iloc[:, 0], best_fit_function)
    test_error = mean_squared_error(test_y, interpolated_values)
    test_errors.append(test_error)

# Step 4: Choose best-performing functions based on test results
best_function_indices = np.argsort(test_errors)[:4]
best_functions = [best_fit_functions[index] for index in best_function_indices]

print("Best-performing functions:")
for i, best_function in enumerate(best_functions):
    print(f"Function {best_function_indices[i] + 1}:")
    print(best_function)

# Plotting the best-fit functions
plt.figure(figsize=(10, 6))
for i, best_fit_function in enumerate(best_fit_functions):
    plt.plot(train_x, best_fit_function, label=f"Ideal Function {best_fit_indices[i]}")
plt.title("Best-fit Ideal Functions")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

# Initialize DataFrame to store mappings and deviations
mapping_df = pd.DataFrame(columns=['X', 'Y', 'Assigned_Function', 'Mapped_Y', 'Deviation'])

# Iterate through each x-y pair in the test dataset
for i in range(len(test)):
    x = test_x[i]
    y_actual = test_y[i]
    
    # Check which of the four chosen ideal functions the x-value belongs to
    assigned_function = None
    for index, function in enumerate(ideal):
        y_mapped = np.interp(x, ideal['x'], ideal[function])
        deviation = abs(y_actual - y_mapped)
        if deviation < 0.01:  # Check if deviation is within a small threshold (adjust as needed)
            assigned_function = index + 1
            break
    
    # If assigned to a function, save mapping and deviation
    if assigned_function is not None:
        mapping_df = mapping_df.append({'X': x, 'Y': y_actual, 'Assigned_Function': assigned_function,
                                        'Mapped_Y': y_mapped, 'Deviation': deviation}, ignore_index=True)

# Save mapping and deviation information to a CSV file
mapping_df.to_csv("mapping_and_deviations.csv", index=False)

# Initialize plots
plt.figure(figsize=(10, 6))

# Plot the test data
plt.scatter(test_x, test_y, color='blue', label='Test Data')

# Plot the chosen ideal functions
for col in ideal.columns:
    plt.plot(ideal['x'], ideal[col], label=f'Ideal Function {col}', linestyle='--')

# Iterate through each x-y pair in the test dataset
for i in range(len(test)):
    x = test_x[i]
    y_actual = test_y[i]
    
    # Check which of the four chosen ideal functions the x-value belongs to
    assigned_function = None
    for index, function in enumerate(ideal):
        y_mapped = np.interp(x, ideal['x'], ideal[function])
        deviation = abs(y_actual - y_mapped)
        if deviation < 0.01:  # Check if deviation is within a small threshold (adjust as needed)
            assigned_function = index + 1
            break
    
    # If assigned to a function, plot the mapped value
    if assigned_function is not None:
        plt.scatter(x, y_mapped, color='red', marker='x')
        mapping_df = mapping_df.append({'X': x, 'Y': y_actual, 'Assigned_Function': assigned_function,
                                        'Mapped_Y': y_mapped, 'Deviation': deviation}, ignore_index=True)

# Set plot labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Test Data and Chosen Ideal Functions')
plt.legend()
plt.grid(True)

# Show plot
plt.show()

# Save mapping and deviation information to a CSV file
mapping_df.to_csv("mapping_and_deviations.csv", index=False)

