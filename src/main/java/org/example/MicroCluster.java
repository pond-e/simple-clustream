package org.example;


public class MicroCluster {
    private long id;
    private int N=1;
    private final double[] LSX;
    private final double[] SSX;
    private long LST;
    private long SST;
    private int m;
    static int tweetVectorSize = 2;

    private final static double EPSILON = 0.00005;
    public static final double MIN_VARIANCE = 1e-50;
    private final double radiusFactor = 1.8;

    public MicroCluster(Tweet tweet, int m, long id) {
        this.LSX = new double[tweetVectorSize];
        this.SSX = new double[tweetVectorSize];
        this.LST = tweet.timestamp;
        this.SST = (long) tweet.timestamp *tweet.timestamp;
        this.m = m;
        this.id = id;

        for (int i = 0; i < tweetVectorSize; i++) {
            this.LSX[i] = tweet.value[i];
            this.SSX[i] = tweet.value[i]*tweet.value[i];

        }
    }

    public void insert(Tweet tweet) {
        N++;
        LST += tweet.timestamp;
        SST += (long) tweet.timestamp *tweet.timestamp;

        for (int i = 0; i < tweetVectorSize; i++) {
            LSX[i] += tweet.value[i];
            SSX[i] += tweet.value[i]*tweet.value[i];
        }
    }

    public void add(MicroCluster other2) {
        N += other2.N;
        this.LST += other2.LST;
        SST += other2.SST;

        for (int i = 0; i < tweetVectorSize; i++) {
            LSX[i] += other2.LSX[i];
            SSX[i] += other2.SSX[i];
        }
    }

    public double[] getCenter() {
        double[] center = new double[tweetVectorSize];
        for (int i = 0; i < tweetVectorSize; i++) {
            center[i] = LSX[i]/N;
        }
        return center;
    }

    public double getRelevanceStamp() {
        if ( N < 2*m )
            return getMuTime();

        return getMuTime() + getSigmaTime() * getQuantile( ((double)m)/(2*N) );
    }

    public long getMuTime() {
        return LST/N;
    }

    public double getSigmaTime() {
        return Math.sqrt((double) SST /N - ((double) LST /N)*((double) LST /N));
    }

    public double getQuantile(double z) {
        return Math.sqrt(2) * inverseError(2*z-1);
    }

    public static double inverseError(double x) {
        double z = Math.sqrt(Math.PI) * x;
        double res = (z) / 2;

        double z2 = z * z;
        double zProd = z * z2; // z^3
        res += (1.0 / 24) * zProd;

        zProd *= z2;  // z^5
        res += (7.0 / 960) * zProd;

        zProd *= z2;  // z^7
        res += (127 * zProd) / 80640;

        zProd *= z2;  // z^9
        res += (4369 * zProd) / 11612160;

        zProd *= z2;  // z^11
        res += (34807 * zProd) / 364953600;

        zProd *= z2;  // z^13
        res += (20036983 * zProd) / 797058662400d;

        return res;
    }

    public double getRadius() {
        if (N == 1) return 0;
        return getDeviation()*radiusFactor;
    }

    public String toString() {
        String res = "";
        res += String.format("N: %d\n", N);
        for(int i = 0; i < LSX.length; i++) {
            res += String.format("LSX[%d]: %f, ", i, LSX[i]);
        }
        res += "\n";
        for(int i = 0; i < SSX.length; i++) {
            res += String.format("SSX[%d]: %f, ", i, SSX[i]);
        }
        res += "\n";
        res += String.format("LST: %d\n", LST);
        res += String.format("SST: %d\n", SST);
        return res;
    }

    private double getDeviation() {
        double[] variance = getVarianceVector();
        double sumOfDeviation = 0.0;
        for (int i = 0; i < variance.length; i++) {
            double d = Math.sqrt(variance[i]);
            sumOfDeviation += d;
        }
        return sumOfDeviation / variance.length;
    }

    private double[] getVarianceVector() {
        double[] res = new double[this.LSX.length];
        for (int i = 0; i < this.LSX.length; i++) {
            double ls = this.LSX[i];
            double ss = this.SSX[i];

            double lsDivN = ls / this.N;
            double lsDivNSquared = lsDivN * lsDivN;
            double ssDivN = ss / this.N;
            res[i] = ssDivN - lsDivNSquared;

           if (res[i] <= 0.0 && res[i] > -EPSILON) {
               res[i] = MIN_VARIANCE;
           }
        }
        return res;
    }
}
