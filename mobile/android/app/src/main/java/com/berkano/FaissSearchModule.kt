package com.berkano

import com.facebook.react.bridge.*
import org.json.JSONArray
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

/**
 * Нативный модуль для работы с векторным поиском через Faiss
 * Данные хранятся в нативной памяти, не попадая в JavaScript heap
 */
class FaissSearchModule(
    private val reactContext: ReactApplicationContext
) : ReactContextBaseJavaModule(reactContext) {

    // Храним данные в нативной памяти
    // Используем MappedByteBuffer вместо FloatArray для больших файлов
    // Это позволяет не загружать все данные в RAM сразу
    private var embeddingsBuffer: java.nio.MappedByteBuffer? = null
    private var captions: List<String>? = null
    private var numVectors: Int = 0
    private var embeddingSize: Int = 512
    private var embeddingsFile: File? = null

    override fun getName(): String {
        return "FaissSearch"
    }

    /**
     * Загружает embeddings из бинарного файла используя memory-mapped файл
     * НЕ загружает все данные в RAM, читает по требованию
     */
    @ReactMethod
    fun loadEmbeddings(embeddingsPath: String, promise: Promise) {
        try {
            val file = File(embeddingsPath)
            if (!file.exists()) {
                promise.reject("FILE_NOT_FOUND", "Embeddings file not found: $embeddingsPath")
                return
            }

            val fileSize = file.length()
            numVectors = (fileSize / 4 / embeddingSize).toInt() // 4 байта на float
            
            // Используем MappedByteBuffer - это НЕ загружает файл в RAM полностью
            // Операционная система управляет памятью и подгружает данные по требованию
            FileInputStream(file).use { fis ->
                val channel = fis.channel
                val byteBuffer = channel.map(
                    java.nio.channels.FileChannel.MapMode.READ_ONLY,
                    0,
                    fileSize
                )
                byteBuffer.order(ByteOrder.LITTLE_ENDIAN)
                embeddingsBuffer = byteBuffer
            }

            embeddingsFile = file

            val result = Arguments.createMap()
            result.putInt("numVectors", numVectors)
            result.putInt("embeddingSize", embeddingSize)
            result.putLong("fileSize", fileSize)

            promise.resolve(result)
            println("Mapped $numVectors vectors of size $embeddingSize from file (${fileSize / 1024 / 1024}MB)")
        } catch (e: OutOfMemoryError) {
            promise.reject("OUT_OF_MEMORY", "Not enough memory to map embeddings file. File too large: ${e.message}", e)
        } catch (e: Exception) {
            promise.reject("LOAD_ERROR", "Failed to load embeddings: ${e.message}", e)
        }
    }

    /**
     * Читает вектор по индексу из memory-mapped файла
     * Не загружает все данные в память
     */
    private fun getVector(index: Int): FloatArray {
        if (embeddingsBuffer == null) {
            throw IllegalStateException("Embeddings not loaded")
        }
        
        val startByte = index * embeddingSize * 4 // 4 байта на float
        val vector = FloatArray(embeddingSize)
        
        embeddingsBuffer!!.position(startByte)
        embeddingsBuffer!!.asFloatBuffer().get(vector)
        
        return vector
    }

    /**
     * Загружает captions из JSON файла
     * Использует streaming парсинг для больших файлов
     */
    @ReactMethod
    fun loadCaptions(captionsPath: String, promise: Promise) {
        try {
            val file = File(captionsPath)
            if (!file.exists()) {
                promise.reject("FILE_NOT_FOUND", "Captions file not found: $captionsPath")
                return
            }

            // Используем BufferedReader для эффективного чтения больших файлов
            val captionsList = mutableListOf<String>()
            var buffer = StringBuilder()
            var inString = false
            var escapeNext = false
            
            BufferedReader(FileReader(file), 8192).use { reader ->
                var char: Int
                while (reader.read().also { char = it } != -1) {
                    when {
                        escapeNext -> {
                            buffer.append(char.toChar())
                            escapeNext = false
                        }
                        char == '\\'.code -> {
                            escapeNext = true
                        }
                        char == '"'.code -> {
                            if (inString) {
                                // Конец строки
                                captionsList.add(buffer.toString())
                                buffer = StringBuilder()
                                inString = false
                            } else {
                                // Начало строки
                                inString = true
                            }
                        }
                        inString -> {
                            buffer.append(char.toChar())
                        }
                        char == ']'.code -> {
                            // Конец массива
                            break
                        }
                    }
                }
            }
            
            captions = captionsList
            val result = Arguments.createMap()
            result.putInt("count", captionsList.size)

            promise.resolve(result)
            println("Loaded ${captionsList.size} captions")
        } catch (e: OutOfMemoryError) {
            promise.reject("OUT_OF_MEMORY", "Not enough memory to load captions. File too large: ${e.message}", e)
        } catch (e: Exception) {
            promise.reject("LOAD_ERROR", "Failed to load captions: ${e.message}", e)
        }
    }


    /**
     * Вычисляет косинусное расстояние между двумя векторами
     */
    private fun cosineDistance(vecA: FloatArray, vecB: FloatArray): Float {
        var dotProduct = 0f
        var normA = 0f
        var normB = 0f

        for (i in vecA.indices) {
            dotProduct += vecA[i] * vecB[i]
            normA += vecA[i] * vecA[i]
            normB += vecB[i] * vecB[i]
        }

        normA = sqrt(normA)
        normB = sqrt(normB)

        if (normA == 0f || normB == 0f) {
            return 1f
        }

        val similarity = dotProduct / (normA * normB)
        return 1f - similarity
    }

    /**
     * Находит топ-K ближайших векторов к запросу
     * Поиск выполняется в нативной памяти, в JS передаются только результаты
     */
    @ReactMethod
    fun search(
        queryEmbedding: ReadableArray,
        topK: Int,
        promise: Promise
    ) {
        try {
            if (embeddingsBuffer == null) {
                promise.reject("NOT_LOADED", "Embeddings not loaded. Call loadEmbeddings first.")
                return
            }

            if (captions == null) {
                promise.reject("NOT_LOADED", "Captions not loaded. Call loadCaptions first.")
                return
            }

            // Конвертируем запрос из ReadableArray в FloatArray
            val query = FloatArray(queryEmbedding.size())
            for (i in 0 until queryEmbedding.size()) {
                query[i] = queryEmbedding.getDouble(i).toFloat()
            }

            if (query.size != embeddingSize) {
                promise.reject("INVALID_SIZE", "Query embedding size ${query.size} != $embeddingSize")
                return
            }

            // Вычисляем расстояния для всех векторов
            // Читаем векторы по требованию из memory-mapped файла
            val distances = mutableListOf<Pair<Int, Float>>()

            for (i in 0 until numVectors) {
                // Читаем вектор из файла по требованию
                val vector = getVector(i)
                val distance = cosineDistance(query, vector)
                distances.add(Pair(i, distance))
            }

            // Сортируем и берем топ-K
            distances.sortBy { it.second }
            val topResults = distances.take(topK)

            // Формируем результаты для передачи в JS
            val results = Arguments.createArray()
            for ((index, distance) in topResults) {
                val resultItem = Arguments.createMap()
                resultItem.putInt("index", index)
                resultItem.putDouble("distance", distance.toDouble())
                resultItem.putString("caption", captions!![index])
                results.pushMap(resultItem)
            }

            promise.resolve(results)
        } catch (e: Exception) {
            promise.reject("SEARCH_ERROR", "Search failed: ${e.message}", e)
        }
    }

    /**
     * Очищает загруженные данные из памяти
     */
    @ReactMethod
    fun clearCache(promise: Promise) {
        embeddingsBuffer = null
        embeddingsFile = null
        captions = null
        numVectors = 0
        promise.resolve(null)
    }

    /**
     * Проверяет, загружены ли данные
     */
    @ReactMethod
    fun isLoaded(promise: Promise) {
        val result = Arguments.createMap()
        result.putBoolean("embeddingsLoaded", embeddingsBuffer != null)
        result.putBoolean("captionsLoaded", captions != null)
        result.putInt("numVectors", numVectors)
        promise.resolve(result)
    }
}

