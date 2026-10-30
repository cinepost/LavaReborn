//! Internal Hydra attribute transport. JSON arrays remain the public scene format.
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use std::fmt;

pub(crate) fn vectors<'de, D, const N: usize>(deserializer: D) -> Result<Vec<[f32; N]>, D::Error>
where
    D: Deserializer<'de>,
    [f32; N]: Deserialize<'de>,
{
    struct V<const N: usize>;
    impl<'de, const N: usize> Visitor<'de> for V<N>
    where
        [f32; N]: Deserialize<'de>,
    {
        type Value = Vec<[f32; N]>;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "float vectors or a packed f32le attribute")
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut output = Vec::new();
            while let Some(value) = seq.next_element()? {
                output.push(value);
            }
            Ok(output)
        }
        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
            let key: String = map
                .next_key()?
                .ok_or_else(|| de::Error::custom("missing f32le payload"))?;
            if key != "f32le" {
                return Err(de::Error::custom("unknown packed attribute encoding"));
            }
            let encoded: String = map.next_value()?;
            if map.next_key::<String>()?.is_some() {
                return Err(de::Error::custom("unexpected packed attribute field"));
            }
            if N == 0 || !encoded.len().is_multiple_of(N * 8) {
                return Err(de::Error::custom("invalid packed attribute length"));
            }
            let nibble = |c: u8| -> Result<u8, A::Error> {
                match c {
                    b'0'..=b'9' => Ok(c - b'0'),
                    b'a'..=b'f' => Ok(c - b'a' + 10),
                    _ => Err(de::Error::custom("invalid packed attribute hex")),
                }
            };
            let mut output = Vec::with_capacity(encoded.len() / (N * 8));
            for vector in encoded.as_bytes().chunks_exact(N * 8) {
                let mut values = [0.; N];
                for (value, hex) in values.iter_mut().zip(vector.as_chunks::<8>().0) {
                    let mut bytes = [0; 4];
                    for (b, pair) in bytes.iter_mut().zip(hex.as_chunks::<2>().0) {
                        *b = (nibble(pair[0])? << 4) | nibble(pair[1])?;
                    }
                    *value = f32::from_le_bytes(bytes);
                }
                output.push(values);
            }
            Ok(output)
        }
    }
    deserializer.deserialize_any(V::<N>)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Deserialize)]
    struct Attribute {
        #[serde(deserialize_with = "vectors")]
        values: Vec<[f32; 3]>,
    }
    #[test]
    fn packed_attributes_preserve_bits_and_accept_legacy_arrays() {
        let values = [[0., -0., 1.], [-0.5, f32::MIN_POSITIVE, f32::MAX]];
        let hex: String = values
            .iter()
            .flatten()
            .flat_map(|v| f32::to_le_bytes(*v))
            .map(|b| format!("{b:02x}"))
            .collect();
        let json = format!(r#"{{"values":{{"f32le":"{hex}"}}}}"#);
        let got: Attribute = serde_json::from_str(&json).unwrap();
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&got.values),
            bytemuck::cast_slice::<_, u8>(&values)
        );
        let legacy: Attribute = serde_json::from_str(r#"{"values":[[0,1,-2]]}"#).unwrap();
        assert_eq!(legacy.values, vec![[0., 1., -2.]]);
        for bad in [
            r#"{"values":{"f32le":"00"}}"#,
            r#"{"values":{"f32le":"gggggggg0000000000000000"}}"#,
            r#"{"values":{"unknown":""}}"#,
            r#"{"values":{"f32le":"","extra":1}}"#,
        ] {
            assert!(
                serde_json::from_str::<Attribute>(bad).is_err(),
                "accepted {bad}"
            );
        }
    }
}
