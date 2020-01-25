### Information to display
- [x] Word frequency over entire set: Wordcloud
- [x] Average reply length per person
- [x] Text frequency against time
- [ ] Most active week message count: what was the maximum messages sent in a week?
- [ ]Least active week message count: What was the minimum messages sent in a week?
- [ ]Average response time per user
- [ ]Weekly average response time per user over time (line graph)
- [ ]Least active week: what week was the user least active?
- [ ]Aggregated text frequency: which day did we send most texts? (Circle plot)
- [ ]Sentiment per person: See [IBM's service](https://cloud.ibm.com/apidocs/natural-language-understanding/natural-language-understanding#sentiment), [Google's serivce](https://cloud.google.com/natural-language/docs/analyzing-sentiment#language-sentiment-string-python). ![](./misc/sentiment_services.jpg)

### Sentiment analysis
This is potentially quite a complicated part of the project. Currently I have a very simple sentiment analysis working which is using the NLTK python library. The main issue I have with this is that it identifies far too many pieces of text as neutral in sentiment, leading to inconclusive results. I would like to connect to an external sentiment analysis service in the future, but this would probably take some time. 

### Useful
![](misc/strftime.png)


