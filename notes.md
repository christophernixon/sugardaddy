## TODO (loosely ordered by importance)
- [x] __Stats:__
    Print simple statistics such as total number of messages per user, most active day, average messages per day, average message length(words and characters), number of times affection is expressed, favourite word, least active week, msg counts for most and least active weeks, average response times.
- [ ] __Graph: General Statistics:__
    A series of bar charts, a bar for each member of chat, a chart for each statistic.
- [ ] __Graph: Most active weekday:__
    Bubble-plot with a bubble for each weekday, size corresponds to number of messages sent on that day.
- [ ] __Graph: Most popular time to text:__
    Bubble-plot with a bubble for each hour of the day (24 hrs) and size is popularity. Otherwise could be done with bar chart.
- [ ] __Add logging:__
    Switch to logging for keeping track of lines processed and important imformation.
- [ ] __Refactor architecture:__
    I think it would be sensible to split the current architecture into two modules:
    1) A data-collection module which reads in the raw chat data, processes it into a dataframe and calculates the general statastics on the chat.
    2) A visualisation module which builds and displays visualisations of the dataframe and general statistics for the given chat.
    This hopefully shouldn't be too hard and it a much more sensible architecture (than one long module)
- [ ] __Graph: Reply length over time:__ 
    A line graph, with a line for each member of the conversation (minimum 2), y-axis reply length, x-axis time since conversation start. [Example.](https://python-graph-gallery.com/124-spaghetti-plot/)
- [ ] __Graph: Message length distribution:__
    Density plot, to get an idea of the distribution of message lengths
- [ ] __Graph: Sentiment against time:__
    Bubble-plot, sentiment on one axis ranging from positive to negative, time on other ranging from start to end of a week, size of bubbles represents message length.
- [x] __Graph: Message against time:__
    Line graph, aggregated message count on y, time on x axis. Allow for differing time bands


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


