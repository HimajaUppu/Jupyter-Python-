#!/usr/bin/env python
# coding: utf-8

# In[3]:


# STEP 1: Import required libraries
# pandas → data manipulation
# numpy → numerical operations
# os → file handling

import pandas as pd
import numpy as np
import os

# Load the fraud dataset
df = pd.read_csv(r"C:\Users\Himaja\Downloads\Fraud Detection Dataset.csv")

# Preview the data
df.head()





# Check number of rows and columns
df.shape


# In[5]:


# Check data types and missing values
df.info()


# In[7]:


# Percentage of missing values per column
df.isnull().mean().sort_values(ascending=False)


# In[8]:


# Standardize column names:
# - remove spaces
# - convert to lowercase
# - replace spaces with underscores

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df.columns


# In[9]:


# Define column roles based on business understanding

id_cols = ["transaction_id", "user_id"]
target_col = ["fraudulent"]

transaction_cols = ["transaction_amount", "transaction_type", "payment_method"]
time_cols = ["time_of_transaction"]
device_geo_cols = ["device_used", "location"]
behavior_cols = [
    "previous_fraudulent_transactions",
    "account_age",
    "number_of_transactions_last_24h"
]


# In[10]:


# Remove rows with missing or invalid transaction amounts
df = df[df["transaction_amount"].notna()]
df = df[df["transaction_amount"] > 0]

# Create a copy to avoid SettingWithCopyWarning
df = df.copy()


# In[12]:


# IDs should be treated as strings and never filled with median/unknown
for c in id_cols:
    df.loc[:, c] = df[c].astype("string").str.strip()


# In[13]:


# Remove rows with missing or duplicate transaction IDs
df = df.dropna(subset=["transaction_id"])
df = df.drop_duplicates(subset=["transaction_id"])

df["transaction_id"].nunique(), df.shape[0]


# In[14]:


# Convert numeric columns safely
numeric_cols = ["transaction_amount"] + time_cols + behavior_cols

for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")


# In[15]:


# Fill missing numeric values with median
df[numeric_cols] = df[numeric_cols].fillna(
    df[numeric_cols].median(numeric_only=True)
)


# In[16]:


# Categorical columns
categorical_cols = ["transaction_type", "payment_method", "device_used", "location"]

for c in categorical_cols:
    df[c] = df[c].astype("string").str.strip().str.lower()
    df[c] = df[c].fillna("unknown")


# In[17]:


# Time must be between 0 and 23
df["time_of_transaction"] = df["time_of_transaction"].round().astype(int)
df = df[df["time_of_transaction"].between(0, 23)]


# In[18]:


# Behavioral rules
df = df[df["account_age"] > 0]
df = df[df["previous_fraudulent_transactions"] >= 0]
df = df[df["number_of_transactions_last_24h"] >= 0]

df = df.copy()


# In[19]:


#What % of transactions are fraud?
total_txns = len(df)
fraud_txns = df["fraudulent"].sum()
fraud_rate = fraud_txns / total_txns * 100

total_txns, fraud_txns, round(fraud_rate, 2)


# In[ ]:


#Do fraud amounts differ from legitimate ones?
df.groupby("fraudulent")["transaction_amount"].agg(
    count="count",
    mean="mean",
    median="median",
    max="max"
)


# In[20]:


#Which payment methods are riskiest?
payment_risk = (
    df.groupby("payment_method")
      .agg(
          total_txns=("fraudulent", "count"),
          fraud_txns=("fraudulent", "sum")
      )
)

payment_risk["fraud_rate_%"] = (
    payment_risk["fraud_txns"] / payment_risk["total_txns"] * 100
)

payment_risk.sort_values("fraud_rate_%", ascending=False)


# In[21]:


#Do certain devices have higher fraud rates?
device_risk = (
    df.groupby("device_used")
      .agg(
          total_txns=("fraudulent", "count"),
          fraud_txns=("fraudulent", "sum")
      )
)

device_risk["fraud_rate_%"] = (
    device_risk["fraud_txns"] / device_risk["total_txns"] * 100
)

device_risk.sort_values("fraud_rate_%", ascending=False)


# In[22]:


#Inspect raw unique values (always do this first)
df["device_used"].value_counts()


# In[23]:


#Standardize device values (canonical mapping)
# Normalize device_used column
df["device_used"] = (
    df["device_used"]
    .astype("string")
    .str.strip()
    .str.lower()
)

# Collapse semantic duplicates
df["device_used"] = df["device_used"].replace({
    "unknown device": "unknown",
    "not available": "unknown",
    "na": "unknown",
    "n/a": "unknown",
    "": "unknown"
})

df["device_used"].value_counts()


# In[24]:


device_risk = (
    df.groupby("device_used")
      .agg(
          total_txns=("fraudulent", "count"),
          fraud_txns=("fraudulent", "sum")
      )
)

device_risk["fraud_rate_%"] = (
    device_risk["fraud_txns"] / device_risk["total_txns"] * 100
)

device_risk.sort_values("fraud_rate_%", ascending=False)


# In[25]:


#Transaction Amount Risk
q1, q2 = df["transaction_amount"].quantile([0.33, 0.66])

df["amount_risk"] = np.select(
    [df["transaction_amount"] <= q1,
     df["transaction_amount"] <= q2],
    ["Low", "Medium"],
    default="High"
)


# In[26]:


#Frequeny Risk
f1, f2 = df["number_of_transactions_last_24h"].quantile([0.33, 0.66])

df["frequency_risk"] = np.select(
    [df["number_of_transactions_last_24h"] <= f1,
     df["number_of_transactions_last_24h"] <= f2],
    ["Low", "Medium"],
    default="High"
)


# In[27]:


#Account Age Risk
a1, a2 = df["account_age"].quantile([0.33, 0.66])

df["account_age_risk"] = np.select(
    [df["account_age"] <= a1,
     df["account_age"] <= a2],
    ["High", "Medium"],
    default="Low"
)


# In[28]:


#Transaction Type Risk
transaction_type_risk_map = {
    "online": "High",
    "transfer": "High",
    "mobile": "Medium",
    "atm": "Medium",
    "pos": "Low",
    "unknown": "Medium"
}

df["transaction_type_risk"] = (
    df["transaction_type"]
    .map(transaction_type_risk_map)
    .fillna("Medium")
)


# In[29]:


#Time Risk + Fraud History
df["time_risk"] = np.where(df["time_of_transaction"] <= 5, "High", "Low")
df["fraud_history_flag"] = (df["previous_fraudulent_transactions"] > 0).astype(int)


# In[30]:


#Final Risk Score & Risk Level
score_map = {"Low": 0, "Medium": 1, "High": 2}

df["risk_score"] = (
    df["amount_risk"].map(score_map) * 2 +
    df["frequency_risk"].map(score_map) * 2 +
    df["account_age_risk"].map(score_map) * 2 +
    df["transaction_type_risk"].map(score_map) * 2 +
    df["fraud_history_flag"] * 3
)

low_cut, high_cut = df["risk_score"].quantile([0.7, 0.9])

df["risk_level"] = np.select(
    [df["risk_score"] <= low_cut,
     df["risk_score"] <= high_cut],
    ["Low", "Medium"],
    default="High"
)

df["risk_level"].value_counts()


# In[31]:


#Investigation Queue
investigation_queue = df[df["risk_level"] == "High"][
    ["transaction_id", "user_id", "transaction_amount",
     "payment_method", "device_used", "risk_score", "risk_level"]
]

investigation_queue.head()


# In[32]:


df.columns = ["_".join(w.capitalize() for w in c.split("_")) for c in df.columns]


# In[35]:


df.columns


# In[38]:


#Identifying Risk Code
df["Risk_Score"].min(), df["Risk_Score"].max()


# In[39]:


df["Risk_Score"].describe()


# In[40]:


df.groupby("Fraudulent")["Risk_Score"].mean()


# In[41]:


df["Risk_Score_Pct"] = (df["Risk_Score"] / 20) * 100


# In[42]:


df.shape


# In[43]:


df.isnull().sum().sort_values(ascending=False)


# In[44]:


df.dtypes


# In[45]:


df["Transaction_Id"].nunique(), df.shape[0]


# In[46]:


df[
    (df["Transaction_Amount"] <= 0) |
    (df["Time_Of_Transaction"] < 0) |
    (df["Time_Of_Transaction"] > 23) |
    (df["Account_Age"] <= 0) |
    (df["Number_Of_Transactions_Last_24h"] < 0)
]


# In[47]:


df["Device_Used"].value_counts()
df["Payment_Method"].value_counts()
df["Transaction_Type"].value_counts()


# In[48]:


df["Fraudulent"].value_counts()


# In[49]:


df["Risk_Score"].describe()


# In[50]:


df["Risk_Level"].value_counts(normalize=True) * 100


# In[51]:


df.sample(5)


# In[52]:


df.describe(include="all")


# In[53]:


#Make Risk bands categorical (memory + clarity)
risk_order = ["Low", "Medium", "High"]

risk_cols = [
    "Amount_Risk",
    "Frequency_Risk",
    "Account_Age_Risk",
    "Transaction_Type_Risk",
    "Time_Risk",
    "Risk_Level"
]

for c in risk_cols:
    df[c] = pd.Categorical(df[c], categories=risk_order, ordered=True)


# In[54]:


df["Data_Ready_Flag"] = 1


# In[55]:


#Which locations have the highest fraud rate?
location_kpi = (
    df.groupby("Location")
      .agg(
          Total_Transactions=("Fraudulent", "count"),
          Fraud_Transactions=("Fraudulent", "sum")
      )
)

location_kpi["Fraud_Rate_%"] = (
    location_kpi["Fraud_Transactions"] / location_kpi["Total_Transactions"] * 100
)

location_kpi.sort_values("Fraud_Rate_%", ascending=False).head(10)


# In[56]:


#Which locations contribute MOST fraud volume?
location_volume_kpi = (
    df.groupby("Location")
      .agg(
          Fraud_Transactions=("Fraudulent", "sum"),
          Fraud_Amount=("Transaction_Amount", lambda x: x[df.loc[x.index, "Fraudulent"] == 1].sum())
      )
      .sort_values("Fraud_Transactions", ascending=False)
)

location_volume_kpi.head(10)


# In[57]:


#Fraud rate by Payment Method (clean view)
payment_kpi = (
    df.groupby("Payment_Method")
      .agg(
          Total_Transactions=("Fraudulent", "count"),
          Fraud_Transactions=("Fraudulent", "sum")
      )
)

payment_kpi["Fraud_Rate_%"] = (
    payment_kpi["Fraud_Transactions"] / payment_kpi["Total_Transactions"] * 100
)

payment_kpi.sort_values("Fraud_Rate_%", ascending=False)


# In[58]:


#Device-based fraud risk (cleaned)
device_kpi = (
    df.groupby("Device_Used")
      .agg(
          Total_Transactions=("Fraudulent", "count"),
          Fraud_Transactions=("Fraudulent", "sum")
      )
)

device_kpi["Fraud_Rate_%"] = (
    device_kpi["Fraud_Transactions"] / device_kpi["Total_Transactions"] * 100
)

device_kpi.sort_values("Fraud_Rate_%", ascending=False)


# In[60]:


#Fraud rate by Risk_Level (VALIDATION KPI)
risk_validation_kpi = (
    df.groupby("Risk_Level", observed=False)
      .agg(
          Total_Transactions=("Fraudulent", "count"),
          Fraud_Transactions=("Fraudulent", "sum")
      )
)

risk_validation_kpi["Fraud_Rate_%"] = (
    risk_validation_kpi["Fraud_Transactions"]
    / risk_validation_kpi["Total_Transactions"] * 100
)

risk_validation_kpi



# In[61]:


#Average fraud amount vs legitimate amount
amount_kpi = df.groupby("Fraudulent")["Transaction_Amount"].agg(
    Avg_Amount="mean",
    Median_Amount="median",
    Max_Amount="max"
)

amount_kpi


# In[62]:


#Top 10 high-risk locations (combined view)
top_high_risk_locations = (
    df[df["Risk_Level"] == "High"]
      .groupby("Location")
      .agg(
          High_Risk_Transactions=("Fraudulent", "count"),
          Fraud_Transactions=("Fraudulent", "sum")
      )
      .sort_values("High_Risk_Transactions", ascending=False)
)

top_high_risk_locations.head(10)


# In[66]:


df.to_csv(
    r"C:\Users\Himaja\Documents\Fraud_Final_Analytical_Dataset2.csv",
    index=False
)


# In[ ]:




