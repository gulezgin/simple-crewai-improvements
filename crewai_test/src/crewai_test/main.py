#!/usr/bin/env python
import sys
import warnings
from crewai_test.crew import CrewaiTest

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def main():
    crew = CrewaiTest()
    print("Tarım Veri Analizi Sistemi")
    print("=" * 50)
    print("Bu sistem, tarım verilerini analiz etmek için üç uzman agent kullanmaktadır:")
    print("1. Veri Kaynakları Uzmanı: En uygun veri kaynaklarını belirler")
    print("2. Veri Analisti: Verileri analiz eder ve içgörüler çıkarır")
    print("3. Rapor Uzmanı: Sonuçları anlaşılır raporlara dönüştürür")
    print("=" * 50)

    while True:
        query = input("\nSorgunuzu girin (çıkmak için 'exit' yazın): ")
        if query.lower() == "exit":
            break

        analysis_depth = input("Analiz derinliği (basic/detailed) [detailed]: ").lower() or "detailed"

        print("\nAnaliz başlatılıyor...")
        print("-" * 50)

        try:
            inputs = {
                'query': query,
                'analysis_depth': analysis_depth
            }
            result = crew.crew().kickoff(inputs=inputs)

            print("\nAnaliz tamamlandı!")
            print("=" * 50)
            print("Sonuçlar:")
            print(result)

        except Exception as e:
            print(f"\nHata oluştu: {str(e)}")
            print("Lütfen tekrar deneyin.")

if __name__ == "__main__":
    main()
