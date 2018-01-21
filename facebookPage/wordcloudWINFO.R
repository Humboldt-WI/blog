# install.packages("wordcloud")
library(wordcloud)

words <- read.csv("wordcloud.csv", sep = ";")
words$words <- words[,1]
words <- words[,-1]



# Plot a simple wordcloud
# wordcloud(words$words, words$weight) 



# To make the clouds look nice, we can add gradient colors using 
# package RColorBrewer
library(RColorBrewer)
# We want a range of colors, a palette, in a hue signifying the 
# positive or negative aspect of the review
bluePalette <- brewer.pal(n = 9, "Blues") # For these palettes, the max is 9


png("wordcloud_packages.png", width=1200,height=800)

# And plot the wordcouds as before.
wordcloud(words$words, words$weight, max.words = 60, 
          colors = bluePalette,scale = c(6.0,0.5),   rot.per=0.45, min.freq=13   )
dev.off()


dic <- read.csv("dic.csv")

#new data cloud
for(i in 1:length(dic[,1])) {
  dic$weight[i] <- round(runif(1, 10, 100),0)
}

str(dic)

dic <- dic %>% 
  rename(word = dimension.reduction) #adjust the name of the first column

png("cloudBigData.png", width=1200,height=800)

redPalette <- brewer.pal(n = 9, "Reds") # For these palettes, the max is 9

wordcloud(words = dic[,1], freq =  dic$weight, max.words = 60, 
          colors = redPalette,scale = c(4.0,0.5),   rot.per=0.25, min.freq=13   )

dev.off()



#neeeeew

social <- read.csv("socialmediawords.csv")

#new data cloud
for(i in 1:length(social[,1])) {
  social$weight[i] <- round(runif(1, 10, 100),0)
}

str(social)

social <- social %>% 
  rename(word = Account) #adjust the name of the first column

png("cloudSocial.png", width=1200,height=800)

greenPalette <- brewer.pal(n = 9, "Greens") # For these palettes, the max is 9

wordcloud(words = social[,1], freq =  social$weight, max.words = 60, 
          colors = greenPalette,scale = c(4.0,0.5),   rot.per=0.25, min.freq=13   )

dev.off()
