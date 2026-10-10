# Mini Model Projesi

Amaç: Sıfırdan küçük bir dil modeli yapıp nasıl çalıştığını öğrenmek.

## Sözlük
- **Model:** İçinde sayılar bulunan bir dosya. Bilgi, sayıların birlikte oluşturduğu dizilişte saklanır.
- **LLM:** Sıradaki kelimeyi/harfi tahmin etmek için eğitilmiş büyük model.
- **Parametre:** Modelin içindeki tablodaki her bir sayı. Başta rastgeledir, eğitimle azar azar değişir.
- **Tokenizer:** Harfleri/kelimeleri sabit numaralara çeviren sözlük. Eğitimle değişmez.
- **Embedding (kart tablosu):** Her numaraya bir sayı kartı veren tablo.
- **Loss (kayıp):** Tahminin ne kadar yanlış olduğunu gösteren tek bir sayı.
- **Gradient (grad):** Her parametre için "bunu artırırsam loss nasıl değişir?" sorusunun cevabı. Eksiyse artırılır, artıysa azaltılır.
- **Optimizer:** Gradient'lere bakarak parametreleri azar azar güncelleyen araç (burada SGD).
- **Learning rate (lr):** Her güncellemede parametrelerin ne kadar değişeceği.
- **Eğitim:** Tahmin et, loss'u ölç, gradient'i hesapla, parametreleri güncelle, tekrarla.
- **Makine öğrenmesi:** Kuralı elle yazmak yerine örneklerden öğrenen sistemler.


## Yaptıklarım
1. Python, klasör, venv ve PyTorch kurulumu
2. Tokenizer: "merhaba" harflerini numaralara çevirdim
3. Parametre tablosu oluşturdum (6 harf x 2 sayı = 12 parametre)
4. Numaralardan kartları çektim, kartlardan her harf için puan ürettim (+18 parametre, toplam 30)
5. Loss hesapladım (başlangıçta yaklaşık 1,8)
6. Her parametre için "bunu biraz artırırsam loss düşer mi?" sorusunun cevabını (gradient) hesaplayıp parametreleri bu yöne azıcık çevirdim, loss düştü
7. Bunu 100 tur döngüye koydum: loss 1,84'ten 0,05'e indi
8. Modele sordum, her harften sonra gelen harfi doğru buldu (`erhabab`)
9. "anaconda" ile denedim, model zorlandı (aşağıda)
10. Modeli önceki harfi de görecek hâle getirdim (iki harfli model), sorun çözüldü

## Deney: "anaconda"
"merhaba" yerine "anaconda" ile denedim. Model 7 tahminden 5'ini doğru buldu, loss 0,47'ye kadar indi ama sıfıra inmedi.

Sebep: Model sadece o anki harfe bakıyor, öncesini hatırlamıyor. "anaconda"da `a`'dan sonra bazen `n` bazen `c` geliyor, ama model iki `a`'yı da aynı karttan gördüğü için ikisine de aynı cevabı veriyor. Aynı durum `n` harfinde de var.

Sonuç: Doğru tahmin için modelin önceki harfleri de görmesi gerekiyor. Gerçek modellerde bunu attention yapıyor.