# ECG_cnn
An implementation of a cnn for abnormal ECG for myocardial infraction 

## 1°/ what's Myocardial infraction :
A heart attack, or myocardial infarction (MI), is permanent damage to the heart muscle. "Myo" means muscle, "cardial" refers to the heart, and "infarction" means death of tissue due to lack of blood supply.

## 2°/ How does that appear in an ECG ?
In a myocardial infarction transmural ischemia develops. In the first hours and days after the onset of a myocardial infarction, several changes can be observed on the ECG. **First, large peaked T waves (or hyperacute T waves), then ST elevation, then negative T waves and finally pathologic Q waves develop.**

> Figure 01 : ECG (normal)
> ![figure10](https://user-images.githubusercontent.com/45218202/130303000-84e25c8b-8038-471c-9e48-f081e1930a37.jpg)

> Figure 02 : ECG (acute MI)
> ![Figure-3](https://user-images.githubusercontent.com/45218202/130303009-d99e0857-b827-48c8-8fc2-8c2a3e7d3fbe.jpg)

## 3°/ Dataset used :
The research paper I relayed on (https://arxiv.org/pdf/1805.00794) considered two different datasets but I worked only with :
     - PTB Diagnostic ECG Dataset : https://archive.physionet.org/physiobank/database/ptbdb/
     *Description of the dataset :* 
           The database contains 549 records from 290 subjects (aged 17 to 87, mean 57.2; 209 men, mean age 55.5, and 81 women, mean age 61.6; ages were not              recorded for 1 female and 14 male subjects). Each subject is represented by one to five records. There are no subjects numbered 124, 132, 134, or              161. Each record includes 15 simultaneously measured signals: the conventional 12 leads (i, ii, iii, avr, avl, avf, v1, v2, v3, v4, v5, v6)                    together with the 3 Frank lead ECGs (vx, vy, vz). Each signal is digitized at 1000 samples per second, with 16 bit resolution over a range of ±                16.384 mV. On special request to the contributors of the database, recordings may be available at sampling rates up to 10 KHz.

## 4°/ CNN model implemented:
> Figure 04 : the proposed architecture of the convolutional neural network 
> 
> ![Capture](https://user-images.githubusercontent.com/45218202/130303416-e6a68e97-a6a0-4f7a-823f-6eec0ad644b6.PNG)

## 5°/ How to use the code and try it :
  1. Download all required packages (preferablly in Anaconda)
     ```
     pip install -r requirements.txt
     ```
  2. Run the python program using : (python3 is used)
     ```
     python main.py
     ```
     
  That's pretty much all !
  
  *If you encounter any problems contact me through mail : il_belkessa@esi.dz*       



