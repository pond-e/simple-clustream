package org.example;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
        CluStream cluster = new CluStream();
        cluster.resetLearning();
        Tweet[] tweets = new Tweet[11];
        double[][] values = {
                {-1, -2},
                {-2, -2},
                {-1, -1},
                {-2, -1},
                {14, 15},
                {14, 16},
                {15, 16},
                {15, 15},
                {5, 2},
                {6, 2}
        };
        for(int i = 0; i < 10; i++) {
            tweets[i] = new Tweet();
            tweets[i].value = values[i];
            tweets[i].timestamp = i+1;
        }

        for(int i = 0; i < 10; i++) {
            cluster.training(tweets[i]);
        }
        MicroCluster[] microClusters = cluster.getMicroClusteringResult();
        for(MicroCluster microCluster : microClusters) {
            System.out.println(microCluster.toString());
        }

    }
}
