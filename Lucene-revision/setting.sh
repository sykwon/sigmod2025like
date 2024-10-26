sudo apt install openjdk-11-jdk
sudo apt install maven

# mvn archetype:generate -DgroupId=com.example.lucene -DartifactId=lucene-demo -DarchetypeArtifactId=maven-archetype-quickstart -DinteractiveMode=false
# cd lucene-demo

# modify pom.xml
# <dependencies>
#     <dependency>
#         <groupId>org.apache.lucene</groupId>
#         <artifactId>lucene-core</artifactId>
#         <version>8.11.1</version>
#     </dependency>
#     <dependency>
#         <groupId>org.apache.lucene</groupId>
#         <artifactId>lucene-queryparser</artifactId>
#         <version>8.11.1</version>
#     </dependency>
# </dependencies>
#
# <properties>
#     <maven.compiler.source>11</maven.compiler.source>
#     <maven.compiler.target>11</maven.compiler.target>
# </properties>
#
# <plugins>
#     <plugin>
#         <groupId>org.apache.maven.plugins</groupId>
#         <artifactId>maven-shade-plugin</artifactId>
#         <version>3.2.4</version>
#         <executions>
#             <execution>
#                 <phase>package</phase>
#                 <goals>
#                     <goal>shade</goal>
#                 </goals>
#             </execution>
#         </executions>
#     </plugin>
# </plugins>
# <build>
#     <plugins>
#         <plugin>
#             <groupId>org.apache.maven.plugins</groupId>
#             <artifactId>maven-shade-plugin</artifactId>
#             <version>3.2.4</version>
#             <executions>
#                 <execution>
#                     <phase>package</phase>
#                     <goals>
#                         <goal>shade</goal>
#                     </goals>
#                     <configuration>
#                         <transformers>
#                             <transformer implementation="org.apache.maven.plugins.shade.resource.ManifestResourceTransformer">
#                                 <mainClass>com.example.lucene.LuceneExample</mainClass>
#                             </transformer>
#                         </transformers>
#                     </configuration>
#                 </execution>
#             </executions>
#         </plugin>
#     </plugins>
# </build>

# mvn complie
