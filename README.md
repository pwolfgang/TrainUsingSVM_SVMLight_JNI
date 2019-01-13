# Train a State Vector Machine

This program trains a State Vector Machine to perform text classification.  The
input is from a database table that contains as a minimum a column containing
the text, a colum containing the classification (an integer), and a column 
containing a unique ID.
This program is based upon the Perl Script run_svm.pl 
<a href="http://www.purpuras.net/pac/run-svm-text.html"> http://www.purpuras.net/pac/run-svm-text.html</a>. 
And implement the algorithm described in Purpura, S., Hillard D. 
<a href="http://www.purpuras.net/dgo2006%20Purpura%20Hillard%20Classifying%20Congressional%20Legislation.pdf">
“Automated Classification of Congressional Legislation.”</a> Proceedings of the Seventh 
International Conference on Digital Government Research. San Diego, CA.

Pair-wise classifiers are trained for each possible pair of categories.

This program takes the following parameters from the command line:

<dl>
<dt>-Xmxnnnnm</dt>
<dd>This is an optional parameter, but if specified it must be first. 
The value nnnn is the number of megabytes of heap space that will be allocated
by the JVM</dd>
<dl>--datasource</dl><dd>The datasource name – see discussion of datasource below</dd>
<dl>--table_name</dl><dd>Table containing the training data</dd>
<dl>--id_column</dl><dd>Column containing the ID</dd>
<dl>--text_column</dl><dd>Column(s) containing the text</dd>
<dl>--code_column</dl><dd>Column containing the code</dd>
<dl>--model</dl><dd>Write the model files to this directory. Default is SVM_Model_Dir</dd>
<dl>--feature_dir</dl><dd>Write the training feature files to this directory
  Default is SVM_Training_Features<dd>
<dl>--use_even</dl><dd>If specified, use even numbered samples for training
  Default is false</dd>
<dl>--use_odd</dl><dd>If if specified, use even numbered samples for training
  Default is false</dd>
<dl>--compute_major</dl><dd>If specified, the major code is computed from the minor code
  Default is false</dd>
<dl>--remove_stopwords [TRUE|FALSE|language]</dl>
<dd>If a language is specified, remove common “stop words” from the text using language specific 
stop words defined obtained from <a href="http://snowball.tartarus.org/dist/snowball_all.tgz">
http://snowball.tartarus.org/dist/snowball_all.tgz </a> If TRUE, 
the stop words are those provided by Chris Buckley and Gerard Salton 
http://www.lextek.com/manuals/onix/stopwords2.html. If FALSE, no stopword filtering
is performed. The default is TRUE</dd>
<dl>--do_stemming [TRUE|FALSE|language]</dl>
<dd>If true, pass all words through the Porter stemmer. If a language is specified
 pass all words through a language-specific stemmer. The language specific 
stemmers were obtained from http://snowball.tartarus.org. The one for 
English is an improvement over Porter’s original.
  Default is true</dd>
<dl>--output_vocab</dl><dd>Optional. If provided the vocabulary is written to the
specified file as a text file</dd>

## DataSource
The datasource is defined by a text file that is used to construct a class that
implements the `java.sql.DataSource` interface. The format of this file is as
follows:
```
jdbc.driver:    *driver class name*
jdbc.url:       *database URL*
jdbc.username:  *database user name*
jdbc.password:  *database password*
```
