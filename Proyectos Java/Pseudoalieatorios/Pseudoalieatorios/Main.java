import java.util.Scanner;
import java.io.IOException;

public class Main {
    public static void main(String[] args) throws IOException, InterruptedException {
        try (Scanner sem = new Scanner(System.in)) {
            String seed, snum2, snum3;
            int tam1, tam2, char1;
            long num1, num2;
            do {
                System.out.println("Escriba el número de la semilla de mínimo 4 dígitos: ");
                seed = sem.next();
                tam1 = seed.length();
                if (tam1 <= 3) {
                    System.out.println("la cantidad de dígitos no es suficiente");
                }
            } while (tam1 <= 3);
            System.out.println("Cantidad de digitos:  " + tam1);
            num1 = Integer.parseInt(seed);

            new ProcessBuilder("cmd", "/c", "cls").inheritIO().start().waitFor();

            for (int i = 1; i <= 10; i++) {
                num2 = (long) Math.pow(num1, 2);
                System.out.print(i + ". (" + num1 + ")^2 = " + num2);
                snum2 = Long.toString(num2);
                tam2 = snum2.length();
                char1 = (tam2 - tam1) / 2;
                snum3 = snum2.substring(char1, char1 + tam1);
                // System.out.println(i + ". " + snum3);
                num1 = Integer.parseInt(snum3);
                Double numR = Double.parseDouble("." + num1);
                System.out.println("\tR1 " + snum3 + "(" + numR + ")");

            }
        } catch (NumberFormatException e) {
            e.printStackTrace();
        }

    }
}
