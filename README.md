# Wine Review Classification
There's so many wines out there today, and often time a trip to the grocery store to pick up a quick drink to go with dinner can take longer than expected. Even when you know the type of wine you like, there's so many different wineries, and let's not even start on the years. Even wanting to try something new can be daunting, what if you spend $15 dollars just to find out you can't stand more than one sip?!
Luckily, there's sommeliers in the world... unluckily, they probably aren't hanging around your local grocery or liquor store. Classification beyond the wine types could prove valuable in this case! Using machine learning and a hefty data source of sommelier's reviews, I plan to perform LSA text classification to find groups of wine, possibly with more in common than just being the same type!
To watch a video presentation of the below topic, please go [here]().
  
## Step 1  
## Exploratory Data Analysis  
Before we start LSA, it can be valuable to begin with an exploration of the dataset being used, even if parts of the dataset won’t be used in the LSA process itself. For full details, see the process [here]( https://github.com/LBusalacchi/msds696/blob/master/1_WineClassification_CleansingandEDA.ipynb). 
Our data looks fairly good, and with some changes to the headers and data types, we are able to dive in. 
  
![alt text](https://github.com/LBusalacchi/msds696/blob/master/Images/1.png)
  
There are lots of countries represented in this list of wine reviews!
  
![alt text](https://github.com/LBusalacchi/msds696/blob/master/Images/2.png)
  
![alt text](https://github.com/LBusalacchi/msds696/blob/master/Images/3.png)
  
There’s some clear leaders when it comes to countries of origin in this dataset, the US, France, and Italy. We can focus on those as we dive into this large dataset (97,850 reviews). 

Overall, we can see some interesting trends on numeric values.   
  
![alt text](https://github.com/LBusalacchi/msds696/blob/master/Images/4.png)   ![alt text](https://github.com/LBusalacchi/msds696/blob/master/Images/5.png)
![alt text](https://github.com/LBusalacchi/msds696/blob/master/Images/6.png)
  
And those are echoed in the top 3 countries as well. 
  
![alt text](https://github.com/LBusalacchi/msds696/blob/master/Images/7.png) 	![alt text](https://github.com/LBusalacchi/msds696/blob/master/Images/8.png)

Have you ever wondered if paying more for a bottle of wine means you’ll get a better quality of wine? Me too! There are no obvious visual correlations going on and running a Pearson’s Correlation test confirms that there isn’t a correlation in the dataset as a whole, as the resultant value is .390. Pearson’s tests indicate a full correlation at 1 or -1, with 0 being no correlation. The standard of .5 or -.5 is the standard for a notable correlation (Brownlee, 2018). 
  
![alt text](https://github.com/LBusalacchi/msds696/blob/master/Images/9.png)
  
An interesting further use of this dataset might be examining whether this correlation is the same at all price points, or if certain price ranges are more likely to be worth your money. 
  
  
## Step 2
## Text Preparation
Before we can begin the LSA process, we need to complete some additional process-specific cleansing on the review text. You can see the full code for this section [here]( https://github.com/LBusalacchi/msds696/blob/master/2_WineClassification_TextPreparation.ipynb).
Our first step is removing the non-alphabetical characters. 
  
Before:
  
![alt text](https://github.com/LBusalacchi/msds696/blob/master/Images/10.png)
  
After:
  
![alt text](https://github.com/LBusalacchi/msds696/blob/master/Images/11.png)
  
Next, we can remove words that are three characters in length or less. 
  
![alt text](https://github.com/LBusalacchi/msds696/blob/master/Images/12.png)
  
We then want to put everything in lowercase letters, so that our later analysis doesn’t think Ripe and ripe are two different words. 
  
![alt text](https://github.com/LBusalacchi/msds696/blob/master/Images/13.png)
  
You could begin the LSA process from here, but initial attempts revealed that there are a lot of wine specific words, such as the names of the types, used within the review text. These are proper nouns in this case, and could unduly influence the topics by creating them by type more so than the other words within the review. Here’s the list of words we’ve removed as a result. 
  
![alt text](https://github.com/LBusalacchi/msds696/blob/master/Images/14.png)
  
  
## Step 3 
## Perform LSA by vectorization and SVD
With all of preparation completed, we can get to the main event, Latent Semantic Analysis, or LSA. You can see the full code [here](https://github.com/LBusalacchi/msds696/blob/master/3_WineClassification_LSA.ipynb).
Our first step is creating the vectorizer, which will create a table of ids by words. We are going to limit this to the top 1000 used words as there are a lot in this data set, but that isn’t necessary for smaller data sets or if computing power is not a concern. We are also going to complete a traditional stop word removal at this point, as words such as “this”, “that”, “from”, etc., won’t be very helpful when creating topics!
  
![alt text](https://github.com/LBusalacchi/msds696/blob/master/Images/15.png)
  
Before we create our model, since we are using the Truncated SVD package from sklearn, we need to define the number of topics we want the model to generate. While some datasets might have an obvious number of topics, ours doesn’t really! You might think the wine variety could be a good place to start, but our dataset is large there 632 distinct groups… while we will end up using a rather large number of topics, 632 is still quite unwieldly. We want our topics to account for as much of the variance as possible, though some recommend that covering 50% of the variance is a good standard (LSA, 2015). We can create some initial models and fit them with different component sizes in order to asses the number of topics we want to create. It’s best to start big here so that we can pare back from there. 
  
![alt text](https://github.com/LBusalacchi/msds696/blob/master/Images/16.png)
  
We will aim for at least the 50% mark for this large of a dataset, which occurs at 219 topics. 
  
![alt text](https://github.com/LBusalacchi/msds696/blob/master/Images/17.png)
  
We can round up a little to create an even number of groups, and use a final component size of 250, which explains 54% of the variance. 
  
![alt text](https://github.com/LBusalacchi/msds696/blob/master/Images/18.png)
  
Now let’s take a look at some of the topics! You can see a dashboard [here]( https://public.tableau.com/profile/lauren.busalacchi#!/vizhome/WineClassifications/Overview) for the full results, and below is a small sample. 
  
![alt text](https://github.com/LBusalacchi/msds696/blob/master/Images/19.png)
  
We can also tie this back to our original dataset (George, 2020b).
  
![alt text](https://github.com/LBusalacchi/msds696/blob/master/Images/20.png)
  
To explore these many topics further, or start looking for a new wine to try, check out this interactive dashboard [here]( https://public.tableau.com/profile/lauren.busalacchi#!/vizhome/WineClassifications/Overview). Variety seems to play little roll within these groups surprisingly enough, my favorite variety of Sauvignon Blanc occurs in over 40 groups! 
  
There’s a lot of potential reasons for this, which gives us many ideas for refining this LSA project in the future. 
  
  * There are a lot of groups, and one of the most obvious methods to pare this down is to pare down the dataset. This could easily be done by price, rating, wine variety, country of origin, or so on. 
  
  * Another way to reduce the groups would be to optimize the number of words our model has to analyze. It’s clear that after looking over the existing topics for awhile that there are more stop words that are highly specific to a wine review that we could remove. We could also lemmatize the words, so that words such as smoke and smoky are considered the same. 

