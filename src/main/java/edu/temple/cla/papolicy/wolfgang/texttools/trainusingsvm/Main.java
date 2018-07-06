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
import tw.edu.ntu.csie.libsvm.svm;
import static tw.edu.ntu.csie.libsvm.svm.svm_train;
import tw.edu.ntu.csie.libsvm.svm_model;
import tw.edu.ntu.csie.libsvm.svm_node;
import tw.edu.ntu.csie.libsvm.svm_parameter;
import tw.edu.ntu.csie.libsvm.svm_problem;

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
        try {
            List<Map<String, Object>> cases = new ArrayList<>();
            Map<String, List<svm_node[]>> trainingSets = new TreeMap<>();
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
                svm_node[] svm_node = Util.convereToSVMNode(attributeSet);
                List<svm_node[]> trainingSet
                        = trainingSets.get(cat);
                if (trainingSet == null) {
                    trainingSet = new ArrayList<>();
                    trainingSets.put(cat, trainingSet);
                }
                trainingSet.add(svm_node);
            });
            buildSVMs(trainingSets, gamma, modelOutput);
            System.err.println("NORMAL COMPLETION");
        } catch (Exception ex) {
            ex.printStackTrace();
        }
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
    public static svm_problem buildTrainingProblem(
            String cat1,
            String cat2,
            Map<String, List<svm_node[]>> trainingSets) {
        List<svm_node[]> trainingSet1 = trainingSets.get(cat1);
        List<svm_node[]> trainingSet2 = trainingSets.get(cat2);
        svm_problem problem = new svm_problem();
        int maxSize = Math.max(trainingSet1.size(), trainingSet2.size());
        problem.l = 2 * maxSize;
        problem.y = new double[2 * maxSize];
        problem.x = new svm_node[2 * maxSize][];
        int j = 0;
        for (int i = 0; i < maxSize; i++) {
            problem.y[j] = 1.0;
            problem.x[j] = trainingSet1.get(i % trainingSet1.size());
            j++;
        }
        for (int i = 0; i < maxSize; i++) {
            problem.y[j] = -1.0;
            problem.x[j] = trainingSet2.get(i % trainingSet2.size());
            j++;
        }
        return problem;
    }

    /**
     * Method to build the svms. Each svm is built by calling svm_learn for each
     * pair of category values.
     *
     * @param trainingSets The training sets
     * @param gamma gamma value for kernel
     * @param modelDir The name of the model directory
     */
    public static void buildSVMs(Map<String, List<svm_node[]>> trainingSets,
            double gamma,
            String modelDir) {
        // Set default svm parameter values.
        svm_parameter param = new svm_parameter();
        param.svm_type = svm_parameter.C_SVC;
        param.kernel_type = svm_parameter.RBF;
        param.degree = 3;
        param.gamma = gamma;	// 1/num_features
        param.coef0 = 0;
        param.nu = 0.5;
        param.cache_size = 100;
        param.C = 1;
        param.eps = 1e-3;
        param.p = 0.1;
        param.shrinking = 1;
        param.probability = 0;
        param.nr_weight = 0;
        param.weight_label = new int[0];
        param.weight = new double[0];

        try {
            File modelDirFile = new File(modelDir);
            String[] cats = trainingSets.keySet().toArray(new String[0]);
            for (int i = 0; i < cats.length - 1; i++) {
                String cat1 = cats[i];
                for (int j = i + 1; j < cats.length; j++) {
                    String cat2 = cats[j];
                    svm_problem problem = buildTrainingProblem(cat1, cat2, trainingSets);
                    System.out.printf("i:%d j:%d%n", i, j);
                    System.out.println("Creating model " + cat1 + "." + cat2);
                    svm_model model = svm_train(problem, param);
                    File modelFile = new File(modelDirFile, "svm." + cat1 + "." + cat2);
                    System.out.println("Writing model " + modelFile.getName());
                    svm.svm_save_model(modelFile.getPath(), model);
                }
            }
        } catch (Exception ex) {
            ex.printStackTrace();
            System.exit(1);
        }
    }
}
