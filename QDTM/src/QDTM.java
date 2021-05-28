import java.io.FileInputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import utils.*;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.stream.IntStream;

import java.util.HashMap;

/**
 * Hierarchical Dirichlet Processes  
 * Chinese Restaurant Franchise Sampler
 * 
 * For more information on the algorithm see:
 * Hierarchical Bayesian Nonparametric Models with Applications. 
 * Y.W. Teh and M.I. Jordan. Bayesian Nonparametrics, 2010. Cambridge University Press.
 * http://www.gatsby.ucl.ac.uk/~ywteh/research/npbayes/TehJor2010a.pdf
 * 
 * For other known implementations see README.txt
 * 
 * @author <a href="mailto:arnim.bleier+hdp@gmail.com">Arnim Bleier</a>
 */



public class QDTM { 


	public double beta  = 0.5; // default only
	public double gamma = 1.5;
	public double alpha = 1.0;
	public double promotion = 0.3;
	public int GPU = 1;
	
	private Random random = new Random();
	private double[] p;
	private double[] f;
	
	protected DOCState[] docStates;
	protected int[] numberOfTablesByTopic;
	protected double[] wordCountByTopic;
	protected double[][] wordCountByTopicAndTerm;
	
	protected int sizeOfVocabulary;
	protected int totalNumberOfWords;
	protected int numberOfTopics;
	protected int totalNumberOfTables;
	
	protected int numberOfQueries;
	protected int[] type_tracker;
	protected List<String> vobs;
	protected int[][] targetByDocAndTerm;
	protected double[][] wordCountByTypeAndTerm;
	protected int[] sizeOfVocabularyByTarget;

	protected HashMap<String, double[]>  wordvectors;
	protected double[][]  similarityMatrix;
	protected ArrayList<ArrayList<Integer>> M;

	double[][] wordTopicProb;
	double[] maxWordTopicProbOverAllTopics;
	double[] maxWordTopicProbOverTargetTopics;
	protected ArrayList<ArrayList<WordState>> queryWords;
	protected int[][] topWordsByTopic;


	public void updateSemanticCoherence(int iter){
		int i, k, n, nIndex;
		double cvik;
		double vb = sizeOfVocabulary * beta;
		wordTopicProb = new double[numberOfTopics][sizeOfVocabulary];
		maxWordTopicProbOverAllTopics = new double[sizeOfVocabulary];
		maxWordTopicProbOverTargetTopics = new double[sizeOfVocabulary];

		for (i = 0; i < sizeOfVocabulary; i++){
			ArrayList<Double> arhCV = new ArrayList<Double>();
			// for (k = numberOfQueries; k < numberOfTopics; k++){
			for (k = 0; k < numberOfTopics; k++){
				if (type_tracker[k] < numberOfQueries){
					cvik = 0.0;
					for (WordState word : queryWords.get(type_tracker[k])){
						nIndex = word.termIndex;
						cvik += (wordCountByTopicAndTerm[k][nIndex] + beta) / (wordCountByTopic[k] + vb) * similarityMatrix[i][nIndex];
					}
					arhCV.add(cvik);
				}
				else{
					cvik = 0.0;
					for (n = 0; n < 10; n++){
						int max_index = topWordsByTopic[k][n];
						if (max_index != -1)
							cvik += (wordCountByTopicAndTerm[k][max_index] + beta) / (wordCountByTopic[k] + vb) * similarityMatrix[i][max_index];
						else
							cvik += beta / (wordCountByTopic[k] + vb) * 0;
					}
					arhCV.add(cvik);
				}
			}

			// map CV to arithmetic CV
			for (n = 0; n < numberOfTopics; n++){
				int max_index = IntStream.range(0, numberOfTopics).boxed().max(Comparator.comparingDouble(ix -> arhCV.get(ix))).get();
				wordTopicProb[max_index][i] = 1.0 - Double.valueOf(n) * 1.0 / Double.valueOf(numberOfTopics - 1);
				arhCV.set(max_index, -10.0);
			}

			for (k = 0; k < numberOfTopics; k++) {
				if (maxWordTopicProbOverAllTopics[i] <= wordTopicProb[k][i]){
					maxWordTopicProbOverAllTopics[i] = wordTopicProb[k][i];
				}
			}
		}
	}

	public void updateGPUFlag(int d, int i, int k){
		DOCState docState = docStates[d];
		int wordIndex = docState.words[i].termIndex;
		double p, u;

		// update GPUFlag
		if (type_tracker[k] >= numberOfQueries || k >= wordTopicProb.length){
			docState.words[i].gpuFlag = 0;
		}
		else{
			p = wordTopicProb[k][wordIndex] / maxWordTopicProbOverAllTopics[wordIndex];
			u = random.nextDouble();
			if (u <= p){
				docState.words[i].gpuFlag = 1;
			}
			else{
				docState.words[i].gpuFlag = 0;
			}
		}
	}

	public void removePromotion(int docID, int i){
		DOCState docState = docStates[docID];
		int table = docState.words[i].tableAssignment;
		int k = docState.tableToTopic[table];
		int type = type_tracker[k];
		int iIndex;

		iIndex = docState.words[i].termIndex;
		if (docState.words[i].gpuFlag == 1 && GPU == 1 && type_tracker[k] < numberOfQueries){	
			wordCountByTopic[k] -= 1.0;
			wordCountByTopicAndTerm[k][iIndex] -= 1.0;
			wordCountByTypeAndTerm[type_tracker[k]][iIndex] -= 1.0;
			docState.wordCountByTable[table] -= 1.0; 
			if (Math.abs(docState.wordCountByTable[table]) < 0.0001) { // table is removed
				totalNumberOfTables --; 
				numberOfTablesByTopic[k]--; 
				docState.tableToTopic[table] --; 
			}

			for (WordState word : queryWords.get(type)){
				int nIndex = word.termIndex;
				if (iIndex == nIndex){
					continue;
				}
				else if(similarityMatrix[iIndex][nIndex] >= 0.5){
					wordCountByTopic[k] -= promotion;
					wordCountByTopicAndTerm[k][nIndex] -= promotion;
					wordCountByTypeAndTerm[type_tracker[k]][nIndex] -= promotion;
					docState.wordCountByTable[table] -= promotion; 
					if (Math.abs(docState.wordCountByTable[table]) < 0.0001) { // table is removed
						totalNumberOfTables -= 1; 
						numberOfTablesByTopic[k] -= 1; 
						docState.tableToTopic[table] --; 
					}
				}
			}
		}
		else{
			wordCountByTopic[k] -= 1; 		
			wordCountByTopicAndTerm[k][iIndex] -= 1;
			wordCountByTypeAndTerm[type_tracker[k]][iIndex] -= 1;
			docState.wordCountByTable[table] -= 1; 
			if (Math.abs(docState.wordCountByTable[table]) < 0.0001) { // table is removed
				totalNumberOfTables--; 
				numberOfTablesByTopic[k]--; 
				docState.tableToTopic[table] --; 
			}
		}
	}

	protected void addPromotion(int docID, int i, int table, int k, int t){
		DOCState docState = docStates[docID]; 
		int iIndex;
		int type = type_tracker[k];
		iIndex = docState.words[i].termIndex;
		docState.words[i].tableAssignment = table; 

		if (docState.words[i].gpuFlag == 1 && GPU == 1 && type_tracker[k] < numberOfQueries){	
			wordCountByTopic[k] += 1.0 ; 
			wordCountByTopicAndTerm[k][iIndex] += 1.0 ; 
			wordCountByTypeAndTerm[t][iIndex] += 1.0 ; 
			docState.wordCountByTable[table] += 1.0 ; 
			if (Math.abs(docState.wordCountByTable[table] - 1.0 ) < 0.0001) { // a new table is created
				docState.numberOfTables++;
				docState.tableToTopic[table] = k;
				totalNumberOfTables++;
				numberOfTablesByTopic[k]++; 
				docState.tableToTopic = ensureCapacity(docState.tableToTopic, docState.numberOfTables);
				docState.wordCountByTable = ensureCapacity(docState.wordCountByTable, docState.numberOfTables);
				if (k == numberOfTopics) { // a new topic is created
					numberOfTopics++; 
					numberOfTablesByTopic = ensureCapacity(numberOfTablesByTopic, numberOfTopics); 
					wordCountByTopic = ensureCapacity(wordCountByTopic, numberOfTopics);
					wordCountByTopicAndTerm = add(wordCountByTopicAndTerm, new double[sizeOfVocabulary], numberOfTopics);
				}
			}

			for (WordState word : queryWords.get(type)){
				int nIndex = word.termIndex;
				if (iIndex == nIndex){
					continue;
				}
				else if (similarityMatrix[iIndex][nIndex] >= 0.5){ 
					wordCountByTopic[k] += promotion; 
					wordCountByTopicAndTerm[k][nIndex] += promotion;
					wordCountByTypeAndTerm[t][nIndex] += promotion;
					docState.wordCountByTable[table] += promotion; 
					if (Math.abs(docState.wordCountByTable[table] - promotion) < 0.0001) { // a new table is created
						docState.numberOfTables++;
						docState.tableToTopic[table] = k;
						totalNumberOfTables++;
						numberOfTablesByTopic[k]++; 
						docState.tableToTopic = ensureCapacity(docState.tableToTopic, docState.numberOfTables);
						docState.wordCountByTable = ensureCapacity(docState.wordCountByTable, docState.numberOfTables);
						if (k == numberOfTopics) { // a new topic is created
							numberOfTopics++; 
							numberOfTablesByTopic = ensureCapacity(numberOfTablesByTopic, numberOfTopics); 
							wordCountByTopic = ensureCapacity(wordCountByTopic, numberOfTopics);
							wordCountByTopicAndTerm = add(wordCountByTopicAndTerm, new double[sizeOfVocabulary], numberOfTopics);
						}
					}
				}
			}
		}
		else{
			wordCountByTopic[k] += 1; 
			wordCountByTopicAndTerm[k][docState.words[i].termIndex] += 1;
			wordCountByTypeAndTerm[t][docState.words[i].termIndex] += 1;
			docState.wordCountByTable[table] += 1; 
			if (Math.abs(docState.wordCountByTable[table] - 1) < 0.0001) { // a new table is created
				docState.numberOfTables++;
				docState.tableToTopic[table] = k;
				totalNumberOfTables++;
				numberOfTablesByTopic[k]++; 
				docState.tableToTopic = ensureCapacity(docState.tableToTopic, docState.numberOfTables);
				docState.wordCountByTable = ensureCapacity(docState.wordCountByTable, docState.numberOfTables);
				if (k == numberOfTopics) { // a new topic is created
					numberOfTopics++; 
					numberOfTablesByTopic = ensureCapacity(numberOfTablesByTopic, numberOfTopics); 
					wordCountByTopic = ensureCapacity(wordCountByTopic, numberOfTopics);
					wordCountByTopicAndTerm = add(wordCountByTopicAndTerm, new double[sizeOfVocabulary], numberOfTopics);
				}
			}
		}
	}

	public void addWordVectors(String path, int V, List<String> vocabulary) throws IOException {
		int i, n;
		int sizeOfVocabulary = V;
		List<String> vobs = vocabulary;
		wordvectors = new HashMap<String, double[]>();

		InputStream is = new FileInputStream(path);
		BufferedReader br = new BufferedReader(new InputStreamReader(is,"UTF-8"));
		String line = null;

		while ((line = br.readLine()) != null) {
			try{
				String[] fields = line.split(" ");
				double[] vector = new double[300];
				for (n = 0; n < 300; n++) {
					vector[n] = Double.parseDouble(fields[n+1]);
				}
				wordvectors.put(fields[0], vector);
			}
			catch (Exception e) {
				System.err.println(e.getMessage() + "\n");
			}
		}

		similarityMatrix = new double[sizeOfVocabulary][sizeOfVocabulary];
		M = new ArrayList<ArrayList<Integer>>();
		for (i = 0; i < sizeOfVocabulary; i++){
			ArrayList<Integer> wSimIndex = new ArrayList<Integer>();
			for (n = 0; n < sizeOfVocabulary; n++){
				if (i == n){
					similarityMatrix[i][n] = 1.;
				}
				else if (wordvectors.get(vobs.get(i)) == null || wordvectors.get(vobs.get(n))==null){
					similarityMatrix[i][n] = 0.;
				}
				else{
					similarityMatrix[i][n] =  cosineSimilarity(wordvectors.get(vobs.get(i)), wordvectors.get(vobs.get(n)));
				}

				if (similarityMatrix[i][n] >= 0.75){
					wSimIndex.add(n);
				}
			}
			M.add(wSimIndex);
			// System.out.println(i + " " + wSimIndex);
		}
	}

	public double cosineSimilarity(double[] vectorA, double[] vectorB) {
		double dotProduct = 0.0;
		double normA = 0.0;
		double normB = 0.0;
		for (int i = 0; i < vectorA.length; i++) {
			dotProduct += vectorA[i] * vectorB[i];
			normA += Math.pow(vectorA[i], 2);
			normB += Math.pow(vectorB[i], 2);
		}   
		return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
	}

	/**
	 * Initially assign the words to tables and topics
	 * 
	 * @param corpus {@link CLDACorpus} on which to fit the model
	 */
	public void addInstances(int[][] documentsInput, int V, int Q, List<String> vocabulary) {
		numberOfQueries	= Q;
		numberOfTopics = Q + 1;
		vobs = vocabulary;

		sizeOfVocabulary = V;
		totalNumberOfWords = 0;
		docStates = new DOCState[documentsInput.length];
		for (int d = 0; d < documentsInput.length; d++) {
			docStates[d] = new DOCState(documentsInput[d], d);
			totalNumberOfWords += documentsInput[d].length;
		}
		int k, i, j;
		DOCState docState;
		p = new double[20]; 
		f = new double[20];
		numberOfTablesByTopic = new int[numberOfTopics+1];
		wordCountByTopic = new  double[numberOfTopics+1];
		wordCountByTopicAndTerm = new double[numberOfTopics+1][];
		for (k = 0; k <= numberOfTopics; k++) 	// var initialization done
			wordCountByTopicAndTerm[k] = new double[sizeOfVocabulary];

		//initialize different queries
		targetByDocAndTerm = new int[documentsInput.length][];
		for (int d = 0; d < documentsInput.length; d++) {	// var initialization done
			targetByDocAndTerm[d] = new int[sizeOfVocabulary];
			for (int q = 0; q < numberOfQueries; q++){
				for (i = 0; i < docStates[q].documentLength; i++){
					targetByDocAndTerm[d][docStates[q].words[i].termIndex] = q + 1;
				}
			}
		}

		queryWords = new ArrayList<ArrayList<WordState>>();
		for (int d = 0; d < numberOfQueries; d++) {
			ArrayList<WordState> words = new ArrayList<WordState>();
			for (int n = 0; n < docStates[d].documentLength; n++){
				int termIndex = docStates[d].words[n].termIndex;
				words.add(new WordState(termIndex, 0, 0));
			}
			queryWords.add(words);
		}

		topWordsByTopic = new int[numberOfTopics][10];

		wordCountByTypeAndTerm = new double[numberOfQueries+2][];
		for (k = 0; k <= numberOfQueries+1; k++) 	// var initialization done
			wordCountByTypeAndTerm[k] = new double[sizeOfVocabulary];
		sizeOfVocabularyByTarget = new int[numberOfQueries+1];

		type_tracker = new int[numberOfTopics];
		for (k = 0; k < numberOfQueries; k++){
			type_tracker[k] = k;
		}

		for (k = numberOfQueries; k < numberOfTopics; k++){
			type_tracker[k] = numberOfQueries;
		} 	// var initialization done
		
		for (j = 0; j < docStates.length; j++) {
			docState = docStates[j]; 
			k = random.nextInt(numberOfTopics - numberOfQueries) + numberOfQueries;
			for (i = 0; i < docState.documentLength; i++) {
				int termIndex = docStates[j].words[i].termIndex;
				if (targetByDocAndTerm[j][termIndex] != 0){
					k = targetByDocAndTerm[j][termIndex] - 1;
					break;
				}
			}
			for (i = 0; i < docState.documentLength; i++) {
				// int termIndex = docStates[j].words[i].termIndex;
				// if (targetByDocAndTerm[j][termIndex] != 0){
				// 	k = targetByDocAndTerm[j][termIndex] - 1;
				// }
				// else{
				// 	k = random.nextInt(numberOfTopics - numberOfQueries) + numberOfQueries;
				// 	// k = random.nextInt(numberOfTopics);
				// }
				addPromotion(docState.docID, i, i, k, type_tracker[k]);
			}
		} // the words in the remaining documents are now assigned too
	}
	
	/**
	 * Step one step ahead
	 * 
	 */
	protected void nextGibbsSweep(int iter) {
		int k, target, table, termIndex;
		updateSemanticCoherence(iter);
		for (int d = 0; d < docStates.length; d++) {
			for (int i = 0; i < docStates[d].documentLength; i++) {
				removePromotion(d, i);
				termIndex = docStates[d].words[i].termIndex;
				if (targetByDocAndTerm[d][termIndex] != 0){
					target = targetByDocAndTerm[d][termIndex] - 1;
					table = sampleTargetTable(d, i, target, 0, iter);
					if (table == docStates[d].numberOfTables){ // new Table
						k = sampleTargetTopic(d, i, target, 0);
					}
					else{
						k = docStates[d].tableToTopic[table];
					}
					updateGPUFlag(d, i, k);
					addPromotion(d, i, table, k, type_tracker[k]);
				}
				else{
					table = sampleTable(d, i, iter);
					if (table == docStates[d].numberOfTables){ // new Table
						k = sampleTopic(d, i);
					}
					else{
						k = docStates[d].tableToTopic[table];
					}
					updateGPUFlag(d, i, k);
					addPromotion(d, i, table, k, type_tracker[k]);
				}
			}
		}
		defragment();
		// print out top words of each topic
		// System.out.println("---------------------------------------");
		topWordsByTopic = ensureCapacity(topWordsByTopic, numberOfTopics);
		for (k = 0; k < numberOfTopics; k++) {
			List<String> word = new ArrayList<String>();
			List<Double> topic_k = new ArrayList<Double>();
			for (double i : wordCountByTopicAndTerm[k])
			{
				topic_k.add(i);
			}
			for (int i = 0; i<10; i++){
				int max_index = IntStream.range(0, sizeOfVocabulary).boxed().max(Comparator.comparingDouble(ix -> topic_k.get(ix))).get();
				if (wordCountByTopicAndTerm[k][max_index] > 0.00001)
					topWordsByTopic[k][i] = max_index;
				else
					topWordsByTopic[k][i] = -1;
				word.add(vobs.get(max_index));
				topic_k.set(max_index, 0.0);
			}
		}
	}

	/**
	 * Step one step ahead
	 * 
	 */
	protected void nextTargetGibbsSweep(int iter) {
		int k, t, table;
		for (int d = 0; d < docStates.length; d++) {
			for (int i = 0; i < docStates[d].documentLength; i++) {
				table = docStates[d].words[i].tableAssignment;
				k = docStates[d].tableToTopic[table];
				t = type_tracker[k];
				if (t < numberOfQueries){
					if (iter == 0)
						removePromotion(d, i);
					else
						removeWord(d, i); // remove the word i from the state
					table = sampleTargetTable(d, i, t, 1, iter);
					if (table == docStates[d].numberOfTables){ // new Table
						k = sampleTargetTopic(d, i, t, 1);
						addWord(d, i, table, k, type_tracker[k]); // sampling its Topic
					}
					else{
						k = docStates[d].tableToTopic[table];
						addWord(d, i, table, k, type_tracker[k]); // existing Table
					}
				}
				else{
					continue;
				}
			}
		}
		defragment();
		// print out top words of each topic
		// System.out.println("---------------------------------------");
		topWordsByTopic = ensureCapacity(topWordsByTopic, numberOfTopics);
		for (k = 0; k < numberOfTopics; k++) {
			List<String> word = new ArrayList<String>();
			List<Double> topic_k = new ArrayList<Double>();
			for (double i : wordCountByTopicAndTerm[k])
			{
				topic_k.add(i);
			}
			for (int i = 0; i<10; i++){
				int max_index = IntStream.range(0, sizeOfVocabulary).boxed().max(Comparator.comparingDouble(ix -> topic_k.get(ix))).get();
				if (wordCountByTopicAndTerm[k][max_index] > 0.00001)
					topWordsByTopic[k][i] = max_index;
				else
					topWordsByTopic[k][i] = -1;
				word.add(vobs.get(max_index));
				topic_k.set(max_index, 0.0);
			}
		}
	}
	
	private void updateSubtopicPriorDensity(){
		sizeOfVocabularyByTarget = new int[numberOfQueries+1];
		for (int t = 0; t <= numberOfQueries; t++){
			for (int i = 0; i < sizeOfVocabulary; i++){
				if (!(wordCountByTypeAndTerm[t][i] < 0.0001)){
					sizeOfVocabularyByTarget[t] ++;
				}
			}
		}
	}

	private int count_vobs(int t){
		return sizeOfVocabularyByTarget[t];
	}


	/**
	 * Decide at which topic the table should be assigned to
	 * 
	 * @param docID the index of the document of the current word
	 * @param i the index of the current word
	 * @param t the index of the target query
	 * @return the index of the topic
	 */
	private int sampleTargetTopic(int docID, int i, int t, int phase) {
		double u, pSum = 0.0;
		int k;
		int sizeOfTargetVocabulary;

		p = ensureCapacity(p, numberOfTopics);
		for (k = 0; k < numberOfTopics; k++) {
			if (type_tracker[k] == t){
				pSum += numberOfTablesByTopic[k] * f[k];
			}
			p[k] = pSum;
		}
		
		if (phase == 0){
			pSum += 0;
		}
		else{
			sizeOfTargetVocabulary = count_vobs(t);
			pSum += gamma / sizeOfTargetVocabulary;
		}

		p[numberOfTopics] = pSum;
		u = random.nextDouble() * pSum;
		for (k = 0; k <= numberOfTopics; k++){
			if (u < p[k])
				break;
		}

		if (k == numberOfTopics){
			type_tracker = ensureCapacity(type_tracker, numberOfTopics);
			type_tracker[numberOfTopics] = t;
		}

		return k;
	}
	
	
		/**	 
	 * Decide at which table the word should be assigned to
	 * 
	 * @param docID the index of the document of the current word
	 * @param i the index of the current word
	 * @param t the index of the target query
	 * @return the index of the table
	 */
	int sampleTargetTable(int docID, int i, int t, int phase, int iter) {	
		int k, j;
		double pSum = 0.0, fNew, u;
		DOCState docState = docStates[docID];
		int total_tables = 0;
		int sizeOfTargetVocabulary;
		double vb ;

		if (phase == 0){
			vb = sizeOfVocabulary * beta;
			fNew = gamma / sizeOfVocabulary;
		}
		else{
			sizeOfTargetVocabulary = count_vobs(t);
			vb = sizeOfTargetVocabulary * beta;
			fNew = gamma / sizeOfTargetVocabulary;
		}

		f = ensureCapacity(f, numberOfTopics);
		p = ensureCapacity(p, docState.numberOfTables);
		
		for (k = 0; k < numberOfTopics; k++) {
			f[k] = (wordCountByTopicAndTerm[k][docState.words[i].termIndex] + beta) / 
					(wordCountByTopic[k] + vb);
			if (phase == 1 && iter == 0 && k < numberOfQueries)
				f[k] = 0;
			if (type_tracker[k] == t){
				total_tables +=	numberOfTablesByTopic[k];
				fNew += numberOfTablesByTopic[k] * f[k];
			}
		}

		for (j = 0; j < docState.numberOfTables; j++) {
			if (Math.abs(docState.wordCountByTable[j]) > 0.0001){
				if (type_tracker[docState.tableToTopic[j]] == t){
					pSum += docState.wordCountByTable[j] * f[docState.tableToTopic[j]];
				}
			}
			p[j] = pSum;
		}

		pSum += alpha * fNew / (total_tables + gamma); // Probability for t = tNew
		p[docState.numberOfTables] = pSum;
		u = random.nextDouble() * pSum;
		for (j = 0; j <= docState.numberOfTables; j++)
			if (u < p[j]) 
				break;	// decided which table the word i is assigned to
		return j;
	}

	/**
	 * Decide at which topic the table should be assigned to
	 * 
	 * @return the index of the topic
	 */
	private int sampleTopic(int docID, int i) {
		double u, pSum = 0.0;
		int k;

		p = ensureCapacity(p, numberOfTopics);
		for (k = 0; k < numberOfTopics; k++) {
			pSum += numberOfTablesByTopic[k] * f[k];
			p[k] = pSum;
		}
		pSum += gamma / sizeOfVocabulary;
		p[numberOfTopics] = pSum;
		
		u = random.nextDouble() * pSum;
		for (k = 0; k <= numberOfTopics; k++){
			if (u < p[k])
				break;
		}

		if (k == numberOfTopics){
			type_tracker = ensureCapacity(type_tracker, numberOfTopics);
			type_tracker[numberOfTopics] = numberOfQueries;
		}
		return k;
	}
	

	/**	 
	 * Decide at which table the word should be assigned to
	 * 
	 * @param docID the index of the document of the current word
	 * @param i the index of the current word
	 * @return the index of the table
	 */
	int sampleTable(int docID, int i, int iter) {	
		int k, j;
		double pSum = 0.0, vb = sizeOfVocabulary * beta, fNew, u;
		DOCState docState = docStates[docID];
		f = ensureCapacity(f, numberOfTopics);
		p = ensureCapacity(p, docState.numberOfTables);
		fNew = gamma / sizeOfVocabulary;
		// fNew = 0.0;
		for (k = 0; k < numberOfTopics; k++) {
			// if (type_tracker[k] == numberOfQueries + 1){
			// 	f[k] = 0.;
			// }
			// else{
				f[k] = (wordCountByTopicAndTerm[k][docState.words[i].termIndex] + beta) / 
							(wordCountByTopic[k] + vb);
			// }
			fNew += numberOfTablesByTopic[k] * f[k];
			
		}

		for (j = 0; j < docState.numberOfTables; j++) {
			if (Math.abs(docState.wordCountByTable[j]) > 0.0001){ 
				pSum += docState.wordCountByTable[j] * f[docState.tableToTopic[j]];
			}
			p[j] = pSum;
		}
		pSum += alpha * fNew / (totalNumberOfTables + gamma); // Probability for t = tNew
		p[docState.numberOfTables] = pSum;
		u = random.nextDouble() * pSum;
		for (j = 0; j <= docState.numberOfTables; j++)
			if (u < p[j]) 
				break;	// decided which table the word i is assigned to
		return j;
	}


	/**
	 * Method to call for fitting the model.
	 * 
	 * @param doShuffle
	 * @param shuffleLag
	 * @param maxIter number of iterations to run
	 * @param saveLag save interval 
	 * @param wordAssignmentsWriter {@link WordAssignmentsWriter}
	 * @param topicsWriter {@link TopicsWriter}
	 * @throws IOException 
	 */
	public void run(int shuffleLag, int maxIter, int subtopicMaxIter, PrintStream log) 
	throws IOException {
		for (int iter = 0; iter < maxIter; iter++) {
			if ((shuffleLag > 0) && (iter > 0) && (iter % shuffleLag == 0))
				doShuffle();
			nextGibbsSweep(iter); 
			log.println("iter = " + iter + " #topics = " + numberOfTopics + ", #tables = "
					+ totalNumberOfTables );
		}

		updateSubtopicPriorDensity();
		
		for (int iter = 0; iter < subtopicMaxIter; iter++) {
			if ((shuffleLag > 0) && (iter > 0) && (iter % shuffleLag == 0))
				doShuffle();
			nextTargetGibbsSweep(iter);
			log.println("iter = " + iter + " #topics = " + numberOfTopics + ", #tables = "
					+ totalNumberOfTables );
		}
	}
		
	
	/**
	 * Removes a word from the bookkeeping
	 * 
	 * @param docID the id of the document the word belongs to 
	 * @param i the index of the word
	 */
	protected void removeWord(int docID, int i){
		DOCState docState = docStates[docID];
		int table = docState.words[i].tableAssignment;
		int k = docState.tableToTopic[table];
		docState.wordCountByTable[table]--; 
		wordCountByTopic[k]--; 		
		wordCountByTopicAndTerm[k][docState.words[i].termIndex] --;
		wordCountByTypeAndTerm[type_tracker[k]][docState.words[i].termIndex] --;
		if (Math.abs(docState.wordCountByTable[table]) < 0.0001) { // table is removed
			totalNumberOfTables--; 
			numberOfTablesByTopic[k]--; 
			docState.tableToTopic[table] --; 
		}
	}
	
	
	
	/**
	 * Add a word to the bookkeeping
	 * 
	 * @param docID	docID the id of the document the word belongs to 
	 * @param i the index of the word
	 * @param table the table to which the word is assigned to
	 * @param k the topic to which the word is assigned to
	 */
	protected void addWord(int docID, int i, int table, int k, int t) {
		DOCState docState = docStates[docID];
		docState.words[i].tableAssignment = table; 
		wordCountByTopic[k]++; 
		wordCountByTopicAndTerm[k][docState.words[i].termIndex] ++;
		wordCountByTypeAndTerm[t][docState.words[i].termIndex] ++;
		docState.wordCountByTable[table] ++; 
		if (Math.abs(docState.wordCountByTable[table] - 1) < 0.0001) { // a new table is created
			docState.numberOfTables++;
			docState.tableToTopic[table] = k;
			totalNumberOfTables++;
			numberOfTablesByTopic[k]++; 
			docState.tableToTopic = ensureCapacity(docState.tableToTopic, docState.numberOfTables);
			docState.wordCountByTable = ensureCapacity(docState.wordCountByTable, docState.numberOfTables);
			if (k == numberOfTopics) { // a new topic is created
				numberOfTopics++; 
				numberOfTablesByTopic = ensureCapacity(numberOfTablesByTopic, numberOfTopics); 
				wordCountByTopic = ensureCapacity(wordCountByTopic, numberOfTopics);
				wordCountByTopicAndTerm = add(wordCountByTopicAndTerm, new double[sizeOfVocabulary], numberOfTopics);
			}
		}
	}

	/**
	 * Removes topics from the bookkeeping that have no words assigned to
	 */
	protected void defragment() {
		int[] kOldToKNew = new int[numberOfTopics];
		int k, newNumberOfTopics = 0;
		for (k = 0; k < numberOfTopics; k++) {
			if (Math.abs(wordCountByTopic[k]) > 0.0001) {
				kOldToKNew[k] = newNumberOfTopics;
				swap(wordCountByTopic, newNumberOfTopics, k);
				swap(numberOfTablesByTopic, newNumberOfTopics, k);
				swap(wordCountByTopicAndTerm, newNumberOfTopics, k);
				swap(type_tracker, newNumberOfTopics, k);
				newNumberOfTopics++;
			} 
		}
		numberOfTopics = newNumberOfTopics;
		for (int j = 0; j < docStates.length; j++) 
			docStates[j].defragment(kOldToKNew);
	}
	
	
	/**
	 * Permute the ordering of documents and words in the bookkeeping
	 */
	protected void doShuffle(){
		List<DOCState> h = Arrays.asList(docStates);
		Collections.shuffle(h);
		docStates = h.toArray(new DOCState[h.size()]);
		for (int j = 0; j < docStates.length; j ++){
			List<WordState> h2 = Arrays.asList(docStates[j].words);
			Collections.shuffle(h2);
			docStates[j].words = h2.toArray(new WordState[h2.size()]);
		}
	}
	
	
	
	public static void swap(int[] arr, int arg1, int arg2){
		int t = arr[arg1]; 
		arr[arg1] = arr[arg2]; 
		arr[arg2] = t; 
	}

	public static void swap(double[] arr, int arg1, int arg2){
		double t = arr[arg1]; 
		arr[arg1] = arr[arg2]; 
		arr[arg2] = t; 
 }
	
	public static void swap(double[][] arr, int arg1, int arg2) {
		double[] t = arr[arg1]; 
		arr[arg1] = arr[arg2]; 
		arr[arg2] = t; 
	}
	
	public static double[] ensureCapacity(double[] arr, int min){
		int length = arr.length;
		if (min < length)
			return arr;
		double[] arr2 = new double[min*2];
		for (int i = 0; i < length; i++) 
			arr2[i] = arr[i];
		return arr2;
	}

	public static int[][] ensureCapacity(int[][] arr, int min){
		int length = arr.length;
		if (min < length)
			return arr;
		int[][] arr2 = new int[min*2][arr[0].length];
		for (int i = 0; i < length; i++) 
			arr2[i] = arr[i];
		return arr2;
	}

	public static int[] ensureCapacity(int[] arr, int min) {
		int length = arr.length;
		if (min < length)
			return arr;
		int[] arr2 = new int[min*2];
		for (int i = 0; i < length; i++) 
			arr2[i] = arr[i];
		return arr2;
	}

	public static double[][] add(double[][] arr, double[] newElement, int index) {
		int length = arr.length;
		if (length <= index){
			double[][] arr2 = new double[index*2][];
			for (int i = 0; i < length; i++) 
				arr2[i] = arr[i];
			arr = arr2;
		}
		arr[index] = newElement;
		return arr;
	}
	
	class DOCState {
		
		int docID, documentLength, numberOfTables;
		int[] tableToTopic; 
	    double[] wordCountByTable;
		WordState[] words;

		public DOCState(int[] instance, int docID) {
			this.docID = docID;
		    numberOfTables = 0;  
		    documentLength = instance.length;
		    words = new WordState[documentLength];	
		    wordCountByTable = new double[2];
		    tableToTopic = new int[2];
			for (int position = 0; position < documentLength; position++) 
				words[position] = new WordState(instance[position], 0, 0);
		}

		public void defragment(int[] kOldToKNew) {
		    int[] tOldToTNew = new int[numberOfTables];
		    int t, newNumberOfTables = 0;
		    for (t = 0; t < numberOfTables; t++){
		        if (Math.abs(wordCountByTable[t]) > 0.0001){
		            tOldToTNew[t] = newNumberOfTables;
		            tableToTopic[newNumberOfTables] = kOldToKNew[tableToTopic[t]];
		            swap(wordCountByTable, newNumberOfTables, t);
		            newNumberOfTables ++;
		        } else 
		        	tableToTopic[t] = -1;
		    }
		    numberOfTables = newNumberOfTables;
		    for (int i = 0; i < documentLength; i++)
		        words[i].tableAssignment = tOldToTNew[words[i].tableAssignment];
		}

	}
	
	
	class WordState {   
	
		int termIndex;
		int tableAssignment;
		int gpuFlag;
		
		public WordState(int wordIndex, int tableAssignment, int gpuFlag){
			this.termIndex = wordIndex;
			this.tableAssignment = tableAssignment;
			this.gpuFlag = gpuFlag;
		}

		public WordState(WordState another) {
			this.termIndex = another.termIndex; // you can access  
			this.tableAssignment = another.tableAssignment; // you can access
			this.gpuFlag = another.gpuFlag; // you can access
		}
	}
	

	public static void main(String[] args) throws IOException {
		String dataset = "20news";
		int querynumber = 16; // number of queries for the 20newsgroup dataset
		// int querynumber = 5; // number of queries for the tagmynews dataset
		// int querynumber = 8; // number of queries for the searchsnippets dataset

		InputStream is = new FileInputStream("../input/vobs_" + dataset + "_test.txt");
		BufferedReader br = new BufferedReader(new InputStreamReader(is,"UTF-8"));
		String line = null;
		List<String> vobs = new ArrayList<String>();
		while ((line = br.readLine()) != null) {
			vobs.add(line);
		}
		QDTM hdp = new QDTM();
		CLDACorpus corpus = new CLDACorpus(new FileInputStream("../input/data_" + dataset + "_test.txt"));

		System.out.println("Loading word embeddings...");
		hdp.addWordVectors("../glove_embedding/glove.6B.300d.txt", corpus.getVocabularySize(), vobs);
		System.out.println("Done");
		hdp.addInstances(corpus.getDocuments(), corpus.getVocabularySize(), querynumber, vobs);

		System.out.println("sizeOfVocabulary = "+hdp.sizeOfVocabulary);
		System.out.println("totalNumberOfWords = "+hdp.totalNumberOfWords);
		System.out.println("NumberOfDocs = "+hdp.docStates.length);

		// sample parent topics
		hdp.run(0, 2000, 0, System.out);
		// PrintStream file = new PrintStream(args[1]);
		PrintStream file = new PrintStream("../results/QDTM-" + dataset + "-parent_nzw.txt");
		for (int k = 0; k < hdp.numberOfTopics; k++) {
			for (int w = 0; w < hdp.sizeOfVocabulary; w++)
				file.format("%05f ",hdp.wordCountByTopicAndTerm[k][w]);
			file.println();
		}
		file.close(); 

		for (int k = 0; k < hdp.numberOfTopics; k++) {
			List<String> word = new ArrayList<String>();
			List<Double> topic_k = new ArrayList<Double>();
			for (double i : hdp.wordCountByTopicAndTerm[k])
			{
				topic_k.add(i);
			}
			for (int i = 0; i<20; i++){
				int max_index = IntStream.range(0, hdp.sizeOfVocabulary).boxed().max(Comparator.comparingDouble(ix -> topic_k.get(ix))).get();
				word.add(vobs.get(max_index));
				topic_k.set(max_index, 0.0);
			}
			System.out.println(hdp.type_tracker[k]);
			System.out.println(word);
		}

		// file = new PrintStream(args[2]);
		file = new PrintStream("../results/QDTM-" + dataset + "-parent.txt");
		file.println("d w z q");
		int t, docID;
		for (int d = 0; d < hdp.docStates.length; d++) {
			DOCState docState = hdp.docStates[d];
			docID = docState.docID;
			for (int i = 0; i < docState.documentLength; i++) {
				t = docState.words[i].tableAssignment;
				file.println(docID + " " + docState.words[i].termIndex + " " + docState.tableToTopic[t] + " " + hdp.type_tracker[docState.tableToTopic[t]]); 
			}
		}
		file.close();
		
		// sample subtopics
		hdp.run(0, 0, 2000, System.out);
		// PrintStream file = new PrintStream(args[1]);
		file = new PrintStream("../results/QDTM-" + dataset + "-sub_nzw.txt");
		for (int k = 0; k < hdp.numberOfTopics; k++) {
			for (int w = 0; w < hdp.sizeOfVocabulary; w++)
				file.format("%05f ",hdp.wordCountByTopicAndTerm[k][w]);
			file.println();
		}
		file.close(); 

		for (int k = 0; k < hdp.numberOfTopics; k++) {
			List<String> word = new ArrayList<String>();
			List<Double> topic_k = new ArrayList<Double>();
			for (double i : hdp.wordCountByTopicAndTerm[k])
			{
				topic_k.add(i);
			}
			for (int i = 0; i<20; i++){
				int max_index = IntStream.range(0, hdp.sizeOfVocabulary).boxed().max(Comparator.comparingDouble(ix -> topic_k.get(ix))).get();
				word.add(vobs.get(max_index));
				topic_k.set(max_index, 0.0);
			}
			System.out.println(hdp.type_tracker[k]);
			System.out.println(word);
		}

		// file = new PrintStream(args[2]);
		file = new PrintStream("../results/QDTM-" + dataset + "-sub.txt");
		file.println("d w z q");
		for (int d = 0; d < hdp.docStates.length; d++) {
			DOCState docState = hdp.docStates[d];
			docID = docState.docID;
			for (int i = 0; i < docState.documentLength; i++) {
				t = docState.words[i].tableAssignment;
				file.println(docID + " " + docState.words[i].termIndex + " " + docState.tableToTopic[t] + " " + hdp.type_tracker[docState.tableToTopic[t]]); 
			}
		}
		file.close();
	}
		
}