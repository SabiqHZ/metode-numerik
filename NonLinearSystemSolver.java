import java.util.ArrayList;
import java.util.List;

public class NonLinearSystemSolver {

    private static final double EPSILON = 0.000001;
    private static final int MAX_ITER = 1000;

    // Fungsi f1(x,y) = x^2 + xy - 10
    public static double f1(double x, double y) {
        return x * x + x * y - 10;
    }

    // Fungsi f2(x,y) = y + 3xy^2 - 57
    public static double f2(double x, double y) {
        return y + 3 * x * y * y - 57;
    }

    // Turunan parsial untuk Jacobian
    public static double df1_dx(double x, double y) {
        return 2 * x + y;
    }

    public static double df1_dy(double x, double y) {
        return x;
    }

    public static double df2_dx(double x, double y) {
        return 3 * y * y;
    }

    public static double df2_dy(double x, double y) {
        return 1 + 6 * x * y;
    }

    // Fungsi iterasi g2A untuk f1: x^2 + xy - 10 = 0 => y = (10 - x^2)/x
    public static double g2A_f1(double x, double y) {
        if (Math.abs(x) < 1e-10) return y; // Hindari pembagian nol
        return (10 - x * x) / x;
    }

    // Fungsi iterasi g1B untuk f2: y + 3xy^2 - 57 = 0 => x = (57 - y)/(3y^2)
    public static double g1B_f2(double x, double y) {
        if (Math.abs(y) < 1e-10) return x; // Hindari pembagian nol
        return (57 - y) / (3 * y * y);
    }

    // Fungsi iterasi g2A untuk f2: y + 3xy^2 - 57 = 0 => y = 57 - 3xy^2
    public static double g2A_f2(double x, double y) {
        return 57 - 3 * x * y * y;
    }

    static class IterationResult {
        int iteration;
        double x;
        double y;
        double f1_val;
        double f2_val;
        double error;

        IterationResult(int iter, double x, double y, double f1, double f2, double err) {
            this.iteration = iter;
            this.x = x;
            this.y = y;
            this.f1_val = f1;
            this.f2_val = f2;
            this.error = err;
        }

        @Override
        public String toString() {
            return String.format("%3d | %12.8f | %12.8f | %12.8f | %12.8f | %12.8e",
                    iteration, x, y, f1_val, f2_val, error);
        }
    }

    // METODE 1: Jacobi dengan g2A dan g1B (NIMx = 2)
    public static List<IterationResult> solveJacobi_g2A_g1B(double x0, double y0) {
        List<IterationResult> results = new ArrayList<>();
        double x = x0, y = y0;

        System.out.println("\n=== METODE 1: JACOBI dengan g2A dan g1B ===");
        System.out.println("Iter |      x       |      y       |     f1       |     f2       |    Error");
        System.out.println("-----|--------------|--------------|--------------|--------------|-------------");

        results.add(new IterationResult(0, x, y, f1(x, y), f2(x, y), 0));
        System.out.println(results.get(0));

        for (int i = 1; i <= MAX_ITER; i++) {
            double x_old = x;
            double y_old = y;

            // Update simultan (Jacobi)
            double y_new = g2A_f1(x_old, y_old);
            double x_new = g1B_f2(x_old, y_old);

            x = x_new;
            y = y_new;

            double error = Math.max(Math.abs(x - x_old), Math.abs(y - y_old));
            IterationResult res = new IterationResult(i, x, y, f1(x, y), f2(x, y), error);
            results.add(res);
            System.out.println(res);

            if (error < EPSILON) {
                System.out.println("\nKonvergen pada iterasi ke-" + i);
                break;
            }
        }

        return results;
    }

    // METODE 2: Seidel dengan g2A dan g2A (NIMx = 2)
    public static List<IterationResult> solveSeidel_g2A_g2A(double x0, double y0) {
        List<IterationResult> results = new ArrayList<>();
        double x = x0, y = y0;

        System.out.println("\n=== METODE 2: GAUSS-SEIDEL dengan g2A dan g2A ===");
        System.out.println("Iter |      x       |      y       |     f1       |     f2       |    Error");
        System.out.println("-----|--------------|--------------|--------------|--------------|-------------");

        results.add(new IterationResult(0, x, y, f1(x, y), f2(x, y), 0));
        System.out.println(results.get(0));

        for (int i = 1; i <= MAX_ITER; i++) {
            double x_old = x;
            double y_old = y;

            // Update sekuensial (Seidel)
            y = g2A_f1(x, y);  // Update y dulu
            x = (57 - y) / (3 * y * y);  // x menggunakan y yang baru (alternatif untuk g2A)

            double error = Math.max(Math.abs(x - x_old), Math.abs(y - y_old));
            IterationResult res = new IterationResult(i, x, y, f1(x, y), f2(x, y), error);
            results.add(res);
            System.out.println(res);

            if (error < EPSILON) {
                System.out.println("\nKonvergen pada iterasi ke-" + i);
                break;
            }
        }

        return results;
    }

    // METODE 3: Newton-Raphson
    public static List<IterationResult> solveNewtonRaphson(double x0, double y0) {
        List<IterationResult> results = new ArrayList<>();
        double x = x0, y = y0;

        System.out.println("\n=== METODE 3: NEWTON-RAPHSON ===");
        System.out.println("Iter |      x       |      y       |     f1       |     f2       |    Error");
        System.out.println("-----|--------------|--------------|--------------|--------------|-------------");

        results.add(new IterationResult(0, x, y, f1(x, y), f2(x, y), 0));
        System.out.println(results.get(0));

        for (int i = 1; i <= MAX_ITER; i++) {
            double x_old = x;
            double y_old = y;

            // Hitung Jacobian
            double j11 = df1_dx(x, y);
            double j12 = df1_dy(x, y);
            double j21 = df2_dx(x, y);
            double j22 = df2_dy(x, y);

            // Hitung determinan
            double det = j11 * j22 - j12 * j21;

            if (Math.abs(det) < 1e-10) {
                System.out.println("Jacobian singular!");
                break;
            }

            // Hitung fungsi
            double f1_val = f1(x, y);
            double f2_val = f2(x, y);

            // Hitung invers Jacobian * F
            double dx = -(j22 * f1_val - j12 * f2_val) / det;
            double dy = -(-j21 * f1_val + j11 * f2_val) / det;

            // Update
            x = x + dx;
            y = y + dy;

            double error = Math.max(Math.abs(dx), Math.abs(dy));
            IterationResult res = new IterationResult(i, x, y, f1(x, y), f2(x, y), error);
            results.add(res);
            System.out.println(res);

            if (error < EPSILON) {
                System.out.println("\nKonvergen pada iterasi ke-" + i);
                break;
            }
        }

        return results;
    }

    // METODE 4: Secant
    public static List<IterationResult> solveSecant(double x0, double y0) {
        List<IterationResult> results = new ArrayList<>();
        double x = x0, y = y0;
        double h = 0.0001; // Perturbasi untuk turunan numerik

        System.out.println("\n=== METODE 4: SECANT ===");
        System.out.println("Iter |      x       |      y       |     f1       |     f2       |    Error");
        System.out.println("-----|--------------|--------------|--------------|--------------|-------------");

        results.add(new IterationResult(0, x, y, f1(x, y), f2(x, y), 0));
        System.out.println(results.get(0));

        for (int i = 1; i <= MAX_ITER; i++) {
            double x_old = x;
            double y_old = y;

            // Aproksimasi Jacobian dengan beda hingga
            double f1_val = f1(x, y);
            double f2_val = f2(x, y);

            double j11 = (f1(x + h, y) - f1_val) / h;
            double j12 = (f1(x, y + h) - f1_val) / h;
            double j21 = (f2(x + h, y) - f2_val) / h;
            double j22 = (f2(x, y + h) - f2_val) / h;

            // Hitung determinan
            double det = j11 * j22 - j12 * j21;

            if (Math.abs(det) < 1e-10) {
                System.out.println("Jacobian singular!");
                break;
            }

            // Hitung invers Jacobian * F
            double dx = -(j22 * f1_val - j12 * f2_val) / det;
            double dy = -(-j21 * f1_val + j11 * f2_val) / det;

            // Update
            x = x + dx;
            y = y + dy;

            double error = Math.max(Math.abs(dx), Math.abs(dy));
            IterationResult res = new IterationResult(i, x, y, f1(x, y), f2(x, y), error);
            results.add(res);
            System.out.println(res);

            if (error < EPSILON) {
                System.out.println("\nKonvergen pada iterasi ke-" + i);
                break;
            }
        }

        return results;
    }

    public static void main(String[] args) {
        double x0 = 1.5;
        double y0 = 3.5;

        System.out.println("PENYELESAIAN SISTEM PERSAMAAN NON-LINEAR");
        System.out.println("f1(x,y) = x^2 + xy - 10 = 0");
        System.out.println("f2(x,y) = y + 3xy^2 - 57 = 0");
        System.out.println("\nNIM: 21120123140058");
        System.out.println("NIMx = 58 mod 4 = 2");
        System.out.println("\nTebakan awal: x0 = " + x0 + ", y0 = " + y0);
        System.out.println("Toleransi (epsilon): " + EPSILON);

        // Jalankan semua metode
        List<IterationResult> jacobi = solveJacobi_g2A_g1B(x0, y0);
        List<IterationResult> seidel = solveSeidel_g2A_g2A(x0, y0);
        List<IterationResult> newton = solveNewtonRaphson(x0, y0);
        List<IterationResult> secant = solveSecant(x0, y0);

        // Ringkasan
        System.out.println("\n\n=== RINGKASAN HASIL ===");
        System.out.println("Metode                    | Iterasi | x akhir      | y akhir      | |f1|         | |f2|");
        System.out.println("--------------------------|---------|--------------|--------------|-------------|-------------");

        IterationResult jLast = jacobi.get(jacobi.size() - 1);
        System.out.printf("Jacobi (g2A+g1B)          | %7d | %12.8f | %12.8f | %11.2e | %11.2e\n",
                jLast.iteration, jLast.x, jLast.y, Math.abs(jLast.f1_val), Math.abs(jLast.f2_val));

        IterationResult sLast = seidel.get(seidel.size() - 1);
        System.out.printf("Seidel (g2A+g2A)          | %7d | %12.8f | %12.8f | %11.2e | %11.2e\n",
                sLast.iteration, sLast.x, sLast.y, Math.abs(sLast.f1_val), Math.abs(sLast.f2_val));

        IterationResult nLast = newton.get(newton.size() - 1);
        System.out.printf("Newton-Raphson            | %7d | %12.8f | %12.8f | %11.2e | %11.2e\n",
                nLast.iteration, nLast.x, nLast.y, Math.abs(nLast.f1_val), Math.abs(nLast.f2_val));

        IterationResult secLast = secant.get(secant.size() - 1);
        System.out.printf("Secant                    | %7d | %12.8f | %12.8f | %11.2e | %11.2e\n",
                secLast.iteration, secLast.x, secLast.y, Math.abs(secLast.f1_val), Math.abs(secLast.f2_val));
    }
}