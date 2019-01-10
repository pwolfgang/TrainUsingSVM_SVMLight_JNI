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

import edu.temple.cla.papolicy.wolfgang.texttools.util.CommonFrontEnd;
import edu.temple.cla.papolicy.wolfgang.texttools.util.Util;
import edu.temple.cla.papolicy.wolfgang.texttools.util.Vocabulary;
import edu.temple.cla.papolicy.wolfgang.texttools.util.WordCounter;
import edu.temple.cla.wolfgang.jnisvmlight.SVMLight;
import java.io.File;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.concurrent.Callable;
import picocli.CommandLine;

/**
 * Create a State Vector Machine to classify text. This program is based upon
 * the Perl Script run_svm.pl
 * <a href="http://www.purpuras.net/pac/run-svm-text.html">
 * http://www.purpuras.net/pac/run-svm-text.html</a>. And implement the
 * algorithm described in Purpura, S., Hillard D.
 * <a href="http://www.purpuras.net/dgo2006%20Purpura%20Hillard%20Classifying%20Congressional%20Legislation.pdf">
 * “Automated Classification of Congressional Legislation.”</a> Proceedings of
 * the Seventh International Conference on Digital Government Research. San
 * Diego, CA.
 *
 * @author Paul Wolfgang
 */
public class Main  implements Callable<Void> {

    @CommandLine.Option(names = "--output_vocab", description = "File where vocabulary is written")
    private String outputVocab;
    
    @CommandLine.Option(names = "--model", description = "Directory where model files are written")
    private String modelOutput = "SVM_Model_Dir";

    private final String[] args;
    private final static SVMLight svmLight = new SVMLight();
    
    public Main(String [] args) {
        this.args = args;
    }

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        Main main = new Main(args);
        CommandLine commandLine = new CommandLine(main);
        commandLine.setUnmatchedArgumentsAllowed(true).parse(args);
        try {
            main.call();
        } catch (Exception ex) {
            throw new RuntimeException(ex);
        }
    }

    @Override
    public Void call() throws Exception {
        long start = System.nanoTime();
        try {
            List<Map<String, Object>> cases = new ArrayList<>();
            Map<String, List<SortedMap<Integer, Double>>> trainingSets = new TreeMap<>();
            CommonFrontEnd commonFrontEnd = new CommonFrontEnd();
            CommandLine commandLine = new CommandLine(commonFrontEnd);
            commandLine.setUnmatchedArgumentsAllowed(true);
            commandLine.parse(args);
            Vocabulary vocabulary = commonFrontEnd.loadData(cases);
            if (outputVocab != null) {
                vocabulary.writeVocabulary(outputVocab);
            }
            double gamma = 1.0 / vocabulary.numFeatures();
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
            cases.forEach(c -> {
                SortedMap<Integer, Double> attributeSet = 
                        Util.computeAttributes((WordCounter)c.get("counts"), vocabulary, gamma);
                String cat = c.get("theCode").toString();
                List<SortedMap<Integer, Double>> trainingSet
                        = trainingSets.get(cat);
                if (trainingSet == null) {
                    trainingSet = new ArrayList<>();
                    trainingSets.put(cat, trainingSet);
                }
                trainingSet.add(attributeSet);
            });
            buildSVMs(trainingSets, modelOutput, vocabulary.numFeatures());
            System.err.println("NORMAL COMPLETION");
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        long end = System.nanoTime();
        System.err.println("TOTAL TIME " + (end-start)/1.0e9 + "sec.");
        System.err.println("SUCESSFUL COMPLETION");
        return null;
    }
    
    /**
     * Method to build training problem.
     *
     * @param cat1 The first category
     * @param cat2 The second category
     * @param trainingSets The attribute data
     * @return A svm_problem
     */
    public static List<SortedMap<Integer, Double>> buildTrainingProblem(
            String cat1,
            String cat2,
            Map<String, List<SortedMap<Integer, Double>>> trainingSets) {
        List<SortedMap<Integer, Double>> trainingSet1 = trainingSets.get(cat1);
        List<SortedMap<Integer, Double>> trainingSet2 = trainingSets.get(cat2);
        int maxSize = Math.max(trainingSet1.size(), trainingSet2.size());
        @SuppressWarnings({"unchecked", "rawtypes"})
        List<SortedMap<Integer, Double>> docs = new ArrayList<>();
        for (int i = 0; i < maxSize; i++) {
            docs.add(trainingSet1.get(i % trainingSet1.size()));
        }
        for (int i = 0; i < maxSize; i++) {
            docs.add(trainingSet2.get(i % trainingSet2.size()));
        }
        return docs;
    }

    /**
     * Method to build the svms. Each svm is built by calling svm_learn for each
     * pair of category values.
     *
     * @param trainingSets The training sets
     * @param modelDir The name of the model directory
     */
    public static void buildSVMs(Map<String, List<SortedMap<Integer, Double>>> trainingSets,
           String modelDir, int totWords) {
        // Set default svm parameter values.
        try {
            File modelDirFile = new File(modelDir);
            String[] cats = trainingSets.keySet().toArray(new String[0]);
            for (int i = 0; i < cats.length - 1; i++) {
                String cat1 = cats[i];
                for (int j = i + 1; j < cats.length; j++) {
                    String cat2 = cats[j];
                    List<SortedMap<Integer, Double>> docs = buildTrainingProblem(cat1, cat2, trainingSets);
                    double[] lables = new double[docs.size()];
                    int halfSize = lables.length/2;
                    for (int k = 0; k < halfSize; k++) {
                        lables[k] = 1;
                    }
                    for (int k = halfSize; k < lables.length; k++) {
                        lables[k] = -1;
                    }
                    System.out.printf("i:%d j:%d%n", i, j);
                    System.out.println("Creating model " + cat1 + "." + cat2);
                    String modelFile = new File(modelDirFile, "svm." + cat1 + "." + cat2).getPath();
                    System.out.println("Writing model " + modelFile);                   
                    svmLight.SVMLearn(docs, lables, docs.size(), totWords, modelFile);
                }
            }
        } catch (Exception ex) {
            ex.printStackTrace();
            System.exit(1);
        }
    }
}
