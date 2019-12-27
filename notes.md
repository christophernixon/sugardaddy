## TODO

- [ ] __Messages with line breaks:__

    Handle texts with line breaks. Currently anything after a line break in a text is skipped.Maybe a cache could be used, where the next line continue to be loaded while a date isn't recognised in them?

    E.g
    ```
    [19/03/2019, 10:11:17] Alice: 😂😂😂😂 beautiful! 
    Well, now you've got the information you need for flights,
    so whenever you decide to buy them or not, you can😬
    ```
- [ ] __Add logging:__
    Switch to logging for keeping track of lines processed and important imformation.
- [ ] __Graph: Reply length over time:__ 
    A line graph, with a line for each member of the conversation (minimum 2), y-axis reply length, x-axis time since conversation start. [Example.](https://python-graph-gallery.com/124-spaghetti-plot/)
- [ ] __Graph: Sentiment against time:__
    Bubble-plot, sentiment on one axis ranging from positive to negative, time on other ranging from start to end of a week, size of bubbles represents message length. 

### Information to display
- Word frequency over entire set: Wordcloud
- Average reply length per person
- Text frequency against time
- Aggregated text frequency: which day did we send most texts? (Circle plot)
- Sentiment per person: See [IBM's service](https://cloud.ibm.com/apidocs/natural-language-understanding/natural-language-understanding#sentiment), [Google's serivce](https://cloud.google.com/natural-language/docs/analyzing-sentiment#language-sentiment-string-python). ![](./misc/sentiment_services.jpg)

