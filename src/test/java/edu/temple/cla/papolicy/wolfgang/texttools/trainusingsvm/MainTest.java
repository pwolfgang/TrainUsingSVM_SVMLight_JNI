/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package edu.temple.cla.papolicy.wolfgang.texttools.trainusingsvm;

import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Paul
 */
public class MainTest {
    
    public MainTest() {
    }

    @Test
    public void testMain() {
        
        String[] args = {"--datasource", "TestDb.txt",
                        "--table_name", "TestTable",
                        "--id_column", "ID",
                        "--text_column", "Abstract",
                        "--code_column", "Code",
                        "--output_vocab", "VocabOut.bin"};
        Main.main(args);
    }
    
}
