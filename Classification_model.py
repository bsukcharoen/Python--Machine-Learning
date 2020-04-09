#!/usr/bin/env python
# coding: utf-8

# In[103]:


#timeit

# Student Name : Pattamaphon Sukcharoen
# Cohort       : 5

# Note: You are only allowed to submit ONE final model for this assignment.


################################################################################
# Import Packages
################################################################################

# use this space for all of your package imports

import pandas                    as        pd               # data science essentials
import matplotlib.pyplot         as        plt              # essential graphical output
import seaborn                   as        sns              # enhanced graphical output
import statsmodels.formula.api   as        smf 
from   sklearn.model_selection   import    train_test_split # train/test split
from   sklearn.ensemble          import    RandomForestClassifier
from   sklearn.metrics           import    roc_auc_score            



################################################################################
# Load Data
################################################################################

# use this space to load the original dataset
# MAKE SURE TO SAVE THE ORIGINAL FILE AS original_df
# Example: original_df = pd.read_excel('Apprentice Chef Dataset.xlsx')

# specifying dataset file name
file = 'Apprentice_Chef_Dataset.xlsx'

# reading the file into Python
chef = pd.read_excel(file)

#In FAMILY_NAME columns ,fill the word "Unknown" to the missing value
chef['FAMILY_NAME'] = chef['FAMILY_NAME'].fillna('Unknown')

################################################################################
# Feature Engineering and (optional) Dataset Standardization
################################################################################

# use this space for all of the feature engineering that is required for your
# final model

# if your final model requires dataset standardization, do this here as well


#___________________________________BINNING__________________________________________
labels_3 = [1,2,3]
#FOLLOWED_RECOMMENDATIONS
chef['BIN_FOLLOWED_RECOMMENDATIONS_PCT'] = pd.cut(chef['FOLLOWED_RECOMMENDATIONS_PCT'],
                                              bins=3,
                                              labels=labels_3)

#___________________________________Trend base__________________________________________

# Feature Engineering (trend changes)                                  

change_cancellations_before_noon_after = 2
change_bin_followed_recommendations_pct_after =2 

chef['change_cancellations_before_noon_after'] = 0
condition = chef.loc[0:,'change_cancellations_before_noon_after'][chef['CANCELLATIONS_BEFORE_NOON'] == change_cancellations_before_noon_after]
chef['change_cancellations_before_noon_after'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

chef['change_bin_followed_recommendations_pct_after'] = 0
condition = chef.loc[0:,'change_bin_followed_recommendations_pct_after'][chef['BIN_FOLLOWED_RECOMMENDATIONS_PCT'] == change_bin_followed_recommendations_pct_after]
chef['change_bin_followed_recommendations_pct_after'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)



#___________________________________One hot encoder__________________________________________

#OHC with emial columns
# placeholder list
placeholder_lst = []

# looping over each email address
for index, col in chef.iterrows():
    
    # splitting email domain at '@'
    split_email = chef.loc[index, 'EMAIL'].split(sep = "@")
    
    # appending placeholder_lst with the results
    placeholder_lst.append(split_email)
    

# converting placeholder_lst into a DataFrame 
email_df = pd.DataFrame(placeholder_lst)

# STEP 2: concatenating with original DataFrame
# renaming column to concatenate
email_df.columns = ['dont_use' , 'email_domain']
print(email_df.columns)


# concatenating personal_email_domain with chef DataFrame
chef = pd.concat([chef, email_df],
                   axis = 1)

# printing value counts of personal_email_domain
chef.loc[: ,'email_domain'].value_counts()

# email domain types
personal_email_domains     = ['@gmail.com','@yahoo.com','@protonmail.com']
professional_email_domains = ['@mmm.com',
                              '@amex.com',
                              '@apple.com',
                              '@boeing.com',
                              '@caterpillar.com',
                              '@chevron.com',
                              '@cisco.com',
                              '@cocacola.com', 
                              '@disney.com', 
                              '@dupont.com' ,
                              '@exxon.com' ,'@ge.or'
                              '@goldmansacs.com' ,
                              '@homedepot.com',
                              '@ibm.com',
                              '@intel.com', 
                              '@jnj.com',
                              '@jpmorgan.com',
                              '@mcdonalds.com',
                              '@merck.com',
                              '@microsoft.com',
                              '@nike.com',
                              '@pfizer.com',
                              '@pg.com',
                              '@travelers.com',
                              '@unitedtech.com',
                              '@unitedhealth.com',
                              '@verizon.com',
                              '@visa.com',
                              '@walmart.com']

Junk_email = ['@me.com',
              '@aol.com',
              '@hotmail.com', 
              '@live.com', 
              '@msn.com',
              '@passport.com']

# placeholder list
placeholder_lst = []

# looping to group observations by domain type
for domain in chef['email_domain']:
        if '@' + domain in personal_email_domains:
            placeholder_lst.append('Personal_eamil')
            
        elif '@' + domain in professional_email_domains:
            placeholder_lst.append('Prof_email')
            
        elif '@' + domain in professional_email_domains:
            placeholder_lst.append('Junk_email')
                                   
        else:
            placeholder_lst.append('Junk_email')


# concatenating with original DataFrame
chef['domain_group'] = pd.Series(placeholder_lst)


# checking results
chef['domain_group'].value_counts()


one_hot_domain_group = pd.get_dummies(chef['domain_group'])
chef = chef.join(one_hot_domain_group)


################################################################################
# Train/Test Split
################################################################################

# use this space to set up testing and validation sets using train/test split

# Note: Be sure to set test_size = 0.25


# declaring explanatory variables
chef_data = chef[['MOBILE_NUMBER',
                 'CANCELLATIONS_BEFORE_NOON',
                 'TASTES_AND_PREFERENCES',
                 'FOLLOWED_RECOMMENDATIONS_PCT',
                 'AVG_CLICKS_PER_VISIT',
                 'BIN_FOLLOWED_RECOMMENDATIONS_PCT',
                 'change_cancellations_before_noon_after',
                 'change_bin_followed_recommendations_pct_after',
                 'Junk_email',
                 'Prof_email']]


# declaring response variable
chef_target = chef['CROSS_SELL_SUCCESS'].astype('int')

# train-test split with stratification
X_train,X_test, y_train, y_test = train_test_split(
            chef_data,
            chef_target,
            test_size = 0.25,
            random_state = 222,
            stratify = chef_target)

# merging training data for statsmodels
chef_train = pd.concat([X_train, y_train], axis = 1)

################################################################################
# Final Model (instantiate, fit, and predict)
################################################################################

# use this space to instantiate, fit, and predict on your final model

rf_tuned = RandomForestClassifier(criterion='gini',
                                           n_estimators=800,
                                           max_depth=6,
                                           min_samples_split= 60,
                                           min_samples_leaf=0.001,
                                           max_features=0.3,
                                           bootstrap= True,
                                           oob_score=True,
                                           random_state=222,
                                           n_jobs=-1) 
rf_tuned.fit(X_train,y_train)

rf_tuned_pred = rf_tuned.predict(X_test)

print('Training ACCURACY:', rf_tuned.score(X_train, y_train).round(4))
print('Testing  ACCURACY:', rf_tuned.score(X_test, y_test).round(4))
print('AUC Score        :', roc_auc_score(y_true  = y_test,
                                          y_score = rf_tuned_pred.round(4)))


################################################################################
# Final Model Score (score)
################################################################################

# use this space to score your final model on the testing set
# MAKE SURE TO SAVE YOUR TEST SCORE AS test_score
# Example: test_score = final_model.score(X_test, y_test)

test_score = roc_auc_score(y_true  = y_test,y_score = rf_tuned_pred.round(4))
print('Test score = ', test_score.round(3))


# In[ ]:




