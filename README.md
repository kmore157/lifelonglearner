# lifelonglearner

Project name: Webpage Classification

Description: When user interacts with a particular website, usually, he\she tries to perform certain actions(click links, submit inputs, etc.) in a typical workflow. As a result of these actions, user is able to visit\interact with different web pages of the website. 
Let's consider a test scenario, where QA needs to validate login functionality. Following would be series of steps involved,
1) visit xyz.com <Tester will manually validate contents and structure of a login webpage>
  i) provide username in username textbox
  ii) provide password in password textbox
  iii) click submit button
2) With correct credentials user should be successfully logged in and navigated to home\dashboard of xyz.com <Again tester will manually validate contents and structure of a home\dashboard webpage>

If we try to automate such workflows of user interactions with the website, first step in the process would be to identify state of current user interaction. i.e. which webpage out of many such webpages of a website, user is currently dealing with. Based on that, corresponding actions could be brought into the picture and performed. 

This project is trying to solve the aforementioned problem statement. We try to build a comprehensive dataset of images\screenshots corresponding to each category, here category is state of user interaction, i.e. webpage out of all webpages of a website.  
We'll use Convolutional Neural Net, on aforementioned dataset and train a model that should be capable of classifying 

Installation: Installation is the next section in an effective README. Tell other users how to install your project locally. Optionally, include a gif to make the process even more clear for other people.

Usage: The next section is usage, in which you instruct other people on how to use your project after they’ve installed it. This would also be a good place to include screenshots of your project in action.

Contributing: 

Credits: Jason Arbon, for inspiration. 

License: TBD
