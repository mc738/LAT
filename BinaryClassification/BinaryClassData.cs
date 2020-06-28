using Microsoft.ML.Data;

namespace LAT.BinaryClassification
{
    public class BinaryClassData
    {
        [LoadColumn(0)]
        public string Value;

        [LoadColumn(1), ColumnName("Label")]
        public bool Rating;
    }
}