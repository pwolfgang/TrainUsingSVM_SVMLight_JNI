/* 
 * Copyright (c) 2018, Temple University
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * All advertising materials features or use of this software must display 
 *   the following  acknowledgement
 *   This product includes software developed by Temple University
 * * Neither the name of the copyright holder nor the names of its 
 *   contributors may be used to endorse or promote products derived 
 *   from this software without specific prior written permission. 
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
package edu.temple.cla.papolicy.wolfgang.texttools.trainusingsvm;

import edu.temple.cla.papolicy.wolfgang.texttools.util.Preprocessor;
import edu.temple.cla.papolicy.wolfgang.texttools.util.Util;
import edu.temple.cla.papolicy.wolfgang.texttools.util.Vocabulary;
import edu.temple.cla.papolicy.wolfgang.texttools.util.WordCounter;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.concurrent.Callable;
import picocli.CommandLine;
import picocli.CommandLine.Option;

/**
 * Create a State Vector Machine to classify text. This program is based upon
 * the Perl Script run_svm.pl <a href="http://www.purpuras.net/pac/run-svm-text.html">
 * http://www.purpuras.net/pac/run-svm-text.html</a>. 
 * And implement the algorithm described in Purpura, S., Hillard D. 
 * <a href="http://www.purpuras.net/dgo2006%20Purpura%20Hillard%20Classifying%20Congressional%20Legislation.pdf">
 * “Automated Classification of Congressional Legislation.”</a> Proceedings of the Seventh 
 * International Conference on Digital Government Research. San Diego, CA.
 * @author Paul Wolfgang
 */
public class Main implements Callable<Void> {
    
    @Option(names = "--datasource", required = true, description = "File containing the datasource properties")
    private String dataSourceFileName;
    
    @Option(names = "--table_name", required = true, description = "The name of the table containing the data")
    private String tableName;

    @Option(names = "--id_column", required = true, description = "Column(s) containing the ID")
    private String idColumn;
    
    @Option(names = "--text_column", required = true, description = "Column(s) containing the text")
    private String textColumn;

    @Option(names = "--code_column", required = true, description = "Column(s) containing the code")
    private String codeColumn;

    @Option(names = "--model", description = "Directory where model files are written")
    private String modelOutput = "SVM_Model_Dir";
    
    @Option(names = "--feature_dir", description = "Directory where feature files are written")
    private String featureDir = "SVM_Training_Features";
    
    @Option(names = "--use_even", description = "Use even numbered samples for training")
    private Boolean useEven = false;
    
    @Option(names = "--use_odd", description = "Use odd numbered samples for training")
    private Boolean useOdd = false;

    @Option(names = "--compute_major", description = "Major code is computed from minor code")
    private Boolean computeMajor = false;

    @Option(names = "--remove_stopwords", description = "Remove common \"stop words\" from the text.")
    private String removeStopWords;

    @Option(names = "--do_stemming", description = "Pass all words through stemming algorithm")
    private String doStemming;

    @Option(names = "--output_vocab", description = "File where vocabulary is written")
    private String outputVocab;

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        CommandLine.call(new Main(), System.err, args);
        
    }    
    
    /**
     * Execute the main program. This method is called after the command line
     * parameters have been populated.
     * @return null.
     */
    @Override
    public Void call() {
        try {
            List<String> ids = new ArrayList<>();
            List<String> ref = new ArrayList<>();
            List<String> lines = new ArrayList<>();
            List<List<String>> text = new ArrayList<>();
            List<WordCounter> counts = new ArrayList<>();
            List<SortedMap<Integer, Double>> attributes = new ArrayList<>();
            Vocabulary vocabulary = new Vocabulary();
            Map<String, List<SortedMap<Integer, Double>>> trainingSets = new TreeMap<>();
            String classPath = System.getProperty("java.class.path");
            System.out.println(classPath);
            Util.readFromDatabase(dataSourceFileName, 
                    tableName, 
                    idColumn, 
                    textColumn, 
                    codeColumn, 
                    computeMajor, 
                    useEven, 
                    useOdd, 
                    ids, 
                    lines, 
                    ref);

            Preprocessor preprocessor = new Preprocessor(doStemming, removeStopWords);
            lines.stream()
                    .map(line -> preprocessor.preprocess(line))
                    .forEach(words -> {
                        WordCounter counter = new WordCounter();
                        words.forEach(word -> {
                            counter.updateCounts(word);
                            vocabulary.updateCounts(word);
                        });
                        counts.add(counter);
            });
            vocabulary.computeProbabilities();
            if (outputVocab != null) {
                vocabulary.writeVocabulary(outputVocab);
            }
            File modelParent = new File(modelOutput);
            Util.delDir(modelParent);
            modelParent.mkdirs();
            File vocabFile = new File(modelParent, "vocab.bin");
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(vocabFile))) {
                oos.writeObject(vocabulary);
            } catch (Exception ex) {
                ex.printStackTrace();
                System.exit(1);
            }
            counts.forEach((counter) -> {
                attributes.add(Util.computeAttributes(counter, vocabulary, 0.0));
            });
            for (int i = 0; i < ref.size(); i++) {
                String cat = ref.get(i);
                SortedMap<Integer, Double> attributeSet = attributes.get(i);
                List<SortedMap<Integer, Double>> trainingSet
                        = trainingSets.get(cat);
                if (trainingSet == null) {
                    trainingSet = new ArrayList<>();
                    trainingSets.put(cat, trainingSet);
                }
                trainingSet.add(attributeSet);
            }
            buildTrainingFiles(featureDir, trainingSets);
            buildSVMs(featureDir, modelOutput);
            System.err.println("NORMAL COMPLETION");
            System.exit(0);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        return null;
    }

    /**
     * Method to build the training files. Each training file has the name
     * train.&lt;cat1&gt;.&lt;cat2&gt; Where &lt;cat1&gt; is the first category,
     * and &lt;cat2&gt; is the second. Each file contains the training sets for
     * &lt;cat1&gt; followed by the training data for &lt;cat2&gt;. The
     * &lt;cat1&gt; records begin with +1 and the &lt;cat2&gt; records begin
     * with -1. Following the +1 or -1 are wordId, attribute value pairs,
     * separated by colons.
     *
     * @param featureDirName The name of the directory where the files are
     * written
     * @param trainingSets The map of categories to attributes
     */
    public static void buildTrainingFiles(
            String featureDirName,
            Map<String, List<SortedMap<Integer, Double>>> trainingSets) {
        String[] cats = trainingSets.keySet().toArray(new String[0]);
        File featureDirFile = new File(featureDirName);
        Util.delDir(featureDirFile);
        featureDirFile.mkdirs();
        for (int i = 0; i < cats.length; i++) {
            for (int j = cats.length - 1; j > i; j--) {
                buildTrainingFile(featureDirFile, cats[i], cats[j], trainingSets);
            }
        }
    }

    /**
     * Method to build training file.
     *
     * @param featureDir The directory where the training file is written
     * @param cat1 The first category
     * @param cat2 The second category
     * @param trainingSets The attribute data
     */
    public static void buildTrainingFile(
            File featureDir,
            String cat1,
            String cat2,
            Map<String, List<SortedMap<Integer, Double>>> trainingSets) {
        try {
            File trainingFile = new File(featureDir, "train." + cat1 + "." + cat2);
            try (PrintWriter out = new PrintWriter(new FileWriter(trainingFile))) {
                List<SortedMap<Integer, Double>> trainingSet1
                        = trainingSets.get(cat1);
                List<SortedMap<Integer, Double>> trainingSet2
                        = trainingSets.get(cat2);
                int maxSize = Math.max(trainingSet1.size(), trainingSet2.size());
//             for (int i = 0; i < maxSize; i++) {
                for (int i = 0; i < trainingSet1.size(); i++) {
                    Util.writeFeatureLine(out, +1, trainingSet1.get(i % trainingSet1.size()));
                }
//             for (int i = 0; i < maxSize; i++) {
                for (int i = 0; i < trainingSet2.size(); i++) {
                    Util.writeFeatureLine(out, -1, trainingSet2.get(i % trainingSet2.size()));
                }
            }
        } catch (Exception ex) {
            ex.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Method to build the svms. Each svm is built by calling svm_learn for each
     * training file in the feature directory.
     *
     * @param featureDir The name of the feature directory
     * @param modelDir The name of the model directory
     */
    public static void buildSVMs(String featureDir, String modelDir) {
        try {
            File featureDirFile = new File(featureDir);
            File modelDirFile = new File(modelDir);
            String[] featureFiles = featureDirFile.list();
            ProcessBuilder pb = new ProcessBuilder();
            for (String featureFile : featureFiles) {
                if (featureFile.startsWith("train.")) {
                    int posDot = featureFile.indexOf('.');
                    String cats = featureFile.substring(posDot);
                    ArrayList<String> command = new ArrayList<>();
                    command.add("svm_learn");
                    command.add(featureDir + "/" + featureFile);
                    command.add(modelDir + "/svm" + cats);
                    File outputFile = new File(modelDir, "temp." + cats);
                    System.out.println(command);
                    pb.command(command);
                    Process p = pb.start();
                    InputStream processOut = p.getInputStream();
                    BufferedReader rdr = new BufferedReader(new InputStreamReader(processOut));
                    try (PrintWriter pwtr = new PrintWriter(new FileWriter(outputFile))) {
                        String line;
                        while ((line = rdr.readLine()) != null) {
                            pwtr.println(line);
                        }
                    }
                    p.waitFor();
                }
            }
        } catch (Exception ex) {
            ex.printStackTrace();
            System.exit(1);
        }
    }
}
