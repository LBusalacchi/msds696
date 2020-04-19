# Wine Review Classification
There's so many wines out there today, and often time a trip to the grocery store to pick up a quick drink to go with dinner can take longer than expected. Even when you know the type of wine you like, there's so many different wineries, and let's not even start on the years. Even wanting to try something new can be daunting, what if you spend $15 dollars just to find out you can't stand more than one sip?!
Luckily, there's sommeliers in the world... unluckily, they probably aren't hanging around your local grocery or liquor store. Classification beyond the wine types could prove valuable in this case! Using machine learning and a hefty data source of sommelier's reviews, I plan to perform LSA text classification to find groups of wine, possibly with more in common than just being the same type!

## Step 1
## Exploratory Data Analysis
Before we start LSA, it can be valuable to begin with an exploration of the dataset being used, even if parts of the dataset won’t be used in the LSA process itself. For full details, see the process **here**. 
Our data looks fairly good, and with some changes to the headers and data types, we are able to dive in. 
**PIC**
There are lots of countries represented in this list of wine reviews!
**pic of reformatted list**
**pic of map**
There’s some clear leaders when it comes to countries of origin in this dataset, the US, France, and Italy. We can focus on those as we dive into this large dataset (97,850 reviews). 
**pic of counts of countries**

Overall, we can see some interesting trends on numeric values. 
**pic of overall boxplots, add formatted summaries too?**
And those are echoed in the top 3 countries as well. 
**pics of boxplots for each country**

Have you ever wondered if paying more for a bottle of wine means you’ll get a better quality of wine? Me too! There are no obvious visual correlations going on, and running a Pearson’s Correlation test confirms that there isn’t a correlation in the dataset as a whole as the resultant value is .390. Pearson’s tests indicate a full correlation at 1 or -1, with 0 being no correlation. The standard of .5 or -.5 is the standard for a notable correlation (Brownlee, 2018). 
**pic of scatter plot**
An interesting further use of this dataset might be examining whether this correlation is the same at all price points, or if certain price ranges are more likely to worth your money. 

## Step 2
## Text Preparation
Before we can begin the LSA process, we need to complete some additional process-specific cleansing on the review text. 
Our first step is removing the non-alphabetical characters. 
**pic of text before and after, use index 0**
Next, we can remove words that are three characters in length or less. 
**pic of text before and after, use index 1**
We then want to put everything in lowercase letters, so that our later analysis doesn’t think Ripe and ripe are two different words. 
**pic of text before and after, use index 0**
You could begin the LSA process from here, but initial attempts revealed that there are a lot of wine specific words, such as the names of the types, used within the review text. These are proper nouns in this case, and could unduly influence the topics by creating them by type more so than the other words within the review. Here’s the list of words we’ve removed as a result. 
**pic of all the words removed in nice table**
## Step 3 
## Perform LSA by vectorization and SVD
With all of preparation completed, we can get to the main event, Latent Semantic Analysis, or LSA. 

