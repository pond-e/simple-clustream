package org.example;

import java.util.LinkedList;
import java.util.List;

public class CluStream {
    private int timeWindow;
    private long timestamp = -1;
    private MicroCluster[] kernels;
    private boolean initialized = false;
    private List<MicroCluster> buffer;
    private Tweet[] bufferTweets;
    private int bufferSize;
    private double t; // kernel RadiFactor
    private int m;
    private final int q = 3; // max num kernels
    static int tweetVectorSize = 2;

    public CluStream() {
    }

    public void resetLearning() {
        this.kernels = new MicroCluster[q];
        // horizon
        int h = 10;
        this.timeWindow = h;
        this.initialized = false;
        this.buffer = new LinkedList<MicroCluster>();
        this.bufferSize = 2 * q;
        this.bufferTweets = new Tweet[bufferSize];
        t = 2;
        m = q; // これでいいのか？
    }

    public void training(Tweet tweet) {
        timestamp = tweet.timestamp;
        if (!initialized) {
            if (buffer.size() < bufferSize) {
                buffer.add(new MicroCluster(tweet, m, timestamp)); // id=timestamp
                bufferTweets[buffer.size() - 1] = tweet;
                return;
            }

            kernels = kMean(buffer, m, bufferSize, bufferTweets);

            buffer.clear();
            initialized = true;
        }

        // 1. Determine closest kernel
        MicroCluster closestKernel = null;
        double minDistance = Double.MAX_VALUE;
        for (MicroCluster kernel : kernels) {
            double distance = distance(tweet.value, kernel.getCenter());
            if (distance < minDistance) {
                closestKernel = kernel;
                minDistance = distance;
            }
        }

        // 2. Check whether tweet fits into closestKernel
        assert closestKernel != null;
        if (closestKernel == null) {
            System.err
                    .println("Error: No closest kernel found for the given tweet. All kernels might be uninitialized.");
            return;
        }
        double radius = closestKernel.getRadius();
        if (minDistance < radius) {
            closestKernel.insert(tweet);
            return;
        }

        // erase old MicroCluster or merge two MicroClusters
        // try to erase old MicroCluster
        long threshold = timestamp - timeWindow;

        for (int i = 0; i < kernels.length; i++) {
            if (kernels[i].getRelevanceStamp() < threshold) {
                kernels[i] = new MicroCluster(tweet, m, timestamp);
                return;
            }
        }

        // do not erase, do merge to MicroClusters
        int closestA = 0;
        int closestB = 0;
        minDistance = Double.MAX_VALUE;
        for (int i = 0; i < kernels.length; i++) {
            double[] centerA = kernels[i].getCenter();
            for (int j = i + 1; j < kernels.length; j++) {
                double dist = distance(centerA, kernels[j].getCenter());
                if (dist < minDistance) {
                    minDistance = dist;
                    closestA = i;
                    closestB = j;
                }
            }
        }

        assert (closestA != closestB);

        kernels[closestA].add(kernels[closestB]);
        kernels[closestB] = new MicroCluster(tweet, m, timestamp);

    }

    public MicroCluster predict(Tweet tweet) {
        MicroCluster closestKernel = null;
        double minDistance = Double.MAX_VALUE;
        for (MicroCluster kernel : kernels) {
            double distance = distance(tweet.value, kernel.getCenter());
            if (distance < minDistance) {
                closestKernel = kernel;
                minDistance = distance;
            }
        }
        assert closestKernel != null;
        return closestKernel;
    }

    public MicroCluster[] kMean(List<MicroCluster> buffer, int k, int bufferSize, Tweet[] tweets) {
        MicroCluster[] kernels = new MicroCluster[k];
        for (int i = 0; i < k; i++) {
            kernels[i] = buffer.get(i);
        }

        boolean kMeanFinishFlag = true;
        while (kMeanFinishFlag) {
            MicroCluster[] newMicroCluster = new MicroCluster[k];

            // buffer[i]に対して一番距離の近いkernels[j]を探して、newMicroClusterで管理
            for (int i = 0; i < bufferSize; i++) {
                int closestKernel = 0;
                double minDistance = Double.MAX_VALUE;
                for (int j = 0; j < k; j++) {
                    double dist = distance(buffer.get(i).getCenter(), kernels[j].getCenter());
                    if (dist < minDistance) {
                        minDistance = dist;
                        closestKernel = j;
                    }
                }
                if (newMicroCluster[closestKernel] == null) {
                    newMicroCluster[closestKernel] = new MicroCluster(tweets[i], m, tweets[i].timestamp);
                } else {
                    newMicroCluster[closestKernel].insert(tweets[i]);
                }
            }

            // 割り当てが同一であればkMeanは終了
            for (int i = 0; i < k; i++) {
                if (newMicroCluster[i] != null) {
                    if (newMicroCluster[i].getCenter() != kernels[i].getCenter()) {
                        kMeanFinishFlag = false;
                    }
                } else {
                    kMeanFinishFlag = true;
                }
            }

            // kernelsをnewMicroClusterに更新
            for (int i = 0; i < k; i++) {
                if (newMicroCluster[i] != null) {
                    kernels[i] = newMicroCluster[i];
                }
            }
        }

        return kernels;
    }

    public MicroCluster[] getMicroClusteringResult() {
        MicroCluster[] res = new MicroCluster[kernels.length];
        for (int i = 0; i < kernels.length; i++) {
            res[i] = kernels[i];
        }

        return res;
    }

    private static double distance(double[] pointA, double[] pointB) {
        double distance = 0.0;
        for (int i = 0; i < pointA.length; i++) {
            double d = pointA[i] - pointB[i];
            distance += d * d;
        }
        return Math.sqrt(distance);
    }
}
