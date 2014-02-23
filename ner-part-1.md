Named Entity Recognition and Pattern Matching - TradeOffs Part 1


This might look familiar to you:
   

         "Regular Expressions: Now You Have Two Problems"


Pattern matching/Regular Expressions are typically used for validation in registration forms. These include emails
and phone numbers most of the time.  

You typically don't have to understand regular expressions to put them to use.

Usually, you can google "email regular expression" and in the first few results you will find what you're looking for.

Copy and paste it and you;re good to go. You might use http://regexpal.com/ to see how well a regular expressions works for
your use case. 

Enter a few inputs, see all yellow, and you're good to go. Sometimes, there might be user complaints
because a regex is too zealous or forgets a certain case. Realistically, this doesn't happen all that often.

That begs the question, what else are regular expressions good for besides registration forms?

The typical programmer, assuming they know regular expressions well enough might use them for grep or find/replace.

This might be in a shell script of some kind if it's somewhat complex. Log files are a huge use case for this.



With that being said, the intrepid adventurer might try to find a more complex concept like a person name.

Let's investigate how to do that with a regular expression. A few things might come up. 

        1. A name begins with a capital letter and is typically followed by another one. Example:

                      ([\w]+\s*){2,}

       2.  What about hyphenated names maybe?  

             ([\w-]+\s*){2,}


      How accurate is this though? Regular expressions have to be very precise in order to make sure they
      don't catch cases they shouldn't. There ar clearly a lot of flaws in this pattern.

   Let's see some stackoverflow links:

     http://stackoverflow.com/questions/275160/regex-for-names
     http://stackoverflow.com/questions/888838/regular-expression-for-validating-names-and-surnames


   More complaining about regex!:
   
     http://forums.devshed.com/regex-programming-147/full-name-string-match-556668.html


 You get the point. This is the problen Named Entity Recognition solves.  

Often times, a question comes up, and unless you've spent some time in Natural Language Processing, it might
seem opaque as to how you might do it. My next entry will go in to the use cases for it, some of the involved
libraries and the tradeoffs of using each. 

Named Entity Recognition definitely comes with its own set of problems. I will demonstrate some of those
next entry as well as discuss where regular expressions can actually be a better solution.
