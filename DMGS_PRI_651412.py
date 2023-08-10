#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt

# Add this line to display the plot within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Read the CSV file
data = pd.read_csv('EyeT_group_dataset_II_image_name_grey_blue_participant_52_trial_0.csv')

# Create a box plot
plt.boxplot(data['Recording timestamp'])  # Replace 'column_name' with the column you want to plot

# Customize labels and title
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Box Plot ')

# Display the plot
plt.show()


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file with a different encoding
data = pd.read_csv('EyeT_group_dataset_II_image_name_grey_blue_participant_52_trial_0.csv', encoding='iso-8859-1')

# Create a bar graph
plt.bar(data['Gaze point Y'], data['Gaze point left X'])
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Graph')
plt.xticks(rotation=45)

# Display the graph inline
plt.show()


# In[66]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Read the CSV file
data = pd.read_csv('EyeT_group_dataset_II_image_name_grey_blue_participant_54_trial_00.csv.csv')

# Print the first few rows of the data to inspect its structure
print(data.head())

# Prepare data
X = data.drop('Eye movement type index', axis=1)  # Features
y = data['Gaze event duration']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if there are missing values in the data
print("Missing values in X_train:", X_train.isnull().sum())

# Create a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# In[36]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Read the CSV file
data = pd.read_csv('EyeT_group_dataset_II_image_name_grey_blue_participant_54_trial_00.csv.csv')

# Prepare data
X = data.drop('Gaze event duration', axis=1)  # Features
y = data['Gaze event duration']  # Target

# Encode categorical variables
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM model
svm_model = SVC(kernel='linear', random_state=42)

# Train the model
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# In[4]:


import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('EyeT_group_dataset_II_image_name_grey_blue_participant_54_trial_0.csv')

# Extract data columns
x = data['Eye movement type index']
y = data['Eye movement type index']

# Create a scatter plot
plt.scatter(x, y)

# Add labels and title
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Scatter Plot')

# Display the plot
plt.show()


# In[56]:


get_ipython().system('pip install wordcloud')
get_ipython().system('pip install matplotlib')


# In[5]:


import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('EyeT_group_dataset_II_image_name_grey_blue_participant_54_trial_0.csv')

# Extract data column
values = data['Gaze event duration']

# Create a histogram
plt.hist(values, bins=10, edgecolor='black')  # Adjust the number of bins as needed

# Add labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram')

# Display the plot
plt.show()


# In[8]:


import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('EyeT_group_dataset_II_image_name_grey_blue_participant_54_trial_0.csv')

# Extract data columns
x = data['Eye movement type index']
y = data['Eye movement type index']

# Create a line plot
plt.plot(x, y)

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Plot')

# Display the plot
plt.show()


# In[9]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('EyeT_group_dataset_II_image_name_grey_blue_participant_54_trial_0.csv')

# If you have duplicate entries in the index column, you can try dropping duplicates
data = data.drop_duplicates(subset=['Recording timestamp', 'Recording duration'])

# Pivot the data for heatmap
pivot_data = data.pivot('Recording timestamp', 'Recording duration', 'Gaze event duration')

# Create a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_data, annot=True, cmap='YlGnBu')  # You can choose a different colormap if needed

# Add title
plt.title('Heatmap from CSV Data')

# Display the plot
plt.show()


# In[ ]:





# In[ ]:




